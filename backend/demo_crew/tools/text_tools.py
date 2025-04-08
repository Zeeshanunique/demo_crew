import os
import re
import urllib.request
import json
from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
from paddleocr import PaddleOCR
from PIL import Image
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

class OCRTool(BaseTool):
    """A tool for extracting text from images using OCR."""
    
    name: str = "OCRTool"
    description: str = "Extracts text from images using Optical Character Recognition"
    ocr: Any = None  # Add this line to declare the ocr field
    
    def __init__(self):
        """Initialize the OCR tool."""
        super().__init__()
        # Initialize PaddleOCR with English as default language
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    def _run(self, image_path: str, lang: str = "en") -> Dict[str, Any]:
        """Extract text from the provided image.
        
        Args:
            image_path: Path to the image file
            lang: Language to use for OCR (default: 'en')
            
        Returns:
            Dictionary containing extracted text
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found at path: {image_path}"}
        
        try:
            # Update OCR language if different from current
            if lang != self.ocr.lang:
                self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            
            # Open the image
            image = Image.open(image_path)
            
            # Perform OCR with PaddleOCR
            result = self.ocr.ocr(image_path, cls=True)
            
            # Process and organize OCR results
            full_text = ""
            words = []
            
            if result:
                for idx, line in enumerate(result[0]):
                    # Each line contains coordinates and text with confidence
                    coords, (text, conf) = line
                    full_text += text + " "
                    
                    # Extract bounding box coordinates
                    bbox = {
                        'x': int(min(coords[0][0], coords[3][0])),
                        'y': int(min(coords[0][1], coords[1][1])),
                        'w': int(abs(coords[1][0] - coords[0][0])),
                        'h': int(abs(coords[2][1] - coords[0][1]))
                    }
                    
                    words.append({
                        'text': text,
                        'conf': float(conf),
                        'bbox': bbox,
                        'line_num': idx,
                        'block_num': 0  # PaddleOCR doesn't provide block info
                    })
            
            return {
                "full_text": full_text.strip(),
                "words": words,
                "language": lang
            }
        except Exception as e:
            return {"error": f"Error performing OCR: {str(e)}"}


class DocumentRAGTool(BaseTool):
    """A tool for searching and retrieving information from documents using RAG."""
    
    name: str = "DocumentRAGTool"
    description: str = "Searches through documents to find relevant information using retrieval-augmented generation"
    documents_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents")
    index: Optional[Any] = None
    embeddings: Any = None
    chunks: List = []
    source_documents: Dict = {}
    
    def __init__(self, documents_dir: str = None):
        """Initialize the document RAG tool.
        
        Args:
            documents_dir: Directory containing documents to index
        """
        super().__init__()
        if documents_dir:
            self.documents_dir = documents_dir
        self.index = None
        self.embeddings = OpenAIEmbeddings()
        self.chunks = []
        self.source_documents = {}
    
    def _run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for information in the indexed documents.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            Dictionary containing search results
        """
        # Ensure index is built
        if not self.index:
            self._build_index()
        
        try:
            # Search the index
            docs_and_scores = self.index.similarity_search_with_score(query, k=top_k)
            
            results = []
            for doc, score in docs_and_scores:
                # Get source info
                source = self.source_documents.get(doc.metadata.get('source', ''), 'Unknown')
                
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'source': source
                })
                
            return {
                'query': query,
                'results': results
            }
        except Exception as e:
            return {'error': f"Error searching documents: {str(e)}"}
    
    def _build_index(self) -> None:
        """Build a searchable index from documents."""
        try:
            if not os.path.exists(self.documents_dir):
                raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")
            
            documents = []
            
            # Process different file types
            for file_path in os.listdir(self.documents_dir):
                full_path = os.path.join(self.documents_dir, file_path)
                
                if not os.path.isfile(full_path):
                    continue
                
                try:
                    # Load documents based on file extension
                    if file_path.lower().endswith('.pdf'):
                        loader = PyPDFLoader(full_path)
                        docs = loader.load()
                    elif file_path.lower().endswith('.txt'):
                        loader = TextLoader(full_path)
                        docs = loader.load()
                    elif file_path.lower().endswith(('.docx', '.doc')):
                        loader = Docx2txtLoader(full_path)
                        docs = loader.load()
                    else:
                        # Skip unsupported file types
                        continue
                    
                    # Store source document info
                    self.source_documents[full_path] = {
                        'filename': file_path,
                        'path': full_path,
                        'type': file_path.split('.')[-1].lower()
                    }
                    
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading document {file_path}: {e}")
            
            if not documents:
                print("No documents found or loaded")
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            self.chunks = text_splitter.split_documents(documents)
            
            # Create a vector index
            try:
                from langchain_community.vectorstores import FAISS
                self.index = FAISS.from_documents(self.chunks, self.embeddings)
                print(f"Successfully indexed {len(self.chunks)} chunks from {len(self.source_documents)} documents")
            except ImportError as e:
                print(f"Error importing FAISS: {e}. Please install with 'pip install faiss-cpu'")
                # Fall back to a simple in-memory search
                self._create_simple_index()
                
        except Exception as e:
            print(f"Error building document index: {e}")

    def _create_simple_index(self):
        """Create a simple in-memory index when FAISS is not available."""
        # This is a very basic fallback when FAISS is not available
        # It won't be as efficient but provides basic functionality
        from collections import defaultdict
        import re
        
        # Create a simple inverted index
        self._inverted_index = defaultdict(list)
        self._documents = []
        
        for i, chunk in enumerate(self.chunks):
            self._documents.append(chunk)
            # Extract words and add to inverted index
            text = chunk.page_content.lower()
            words = re.findall(r'\b\w+\b', text)
            for word in set(words):  # Use set to count each word only once per document
                self._inverted_index[word].append(i)


class WebSearchTool(BaseTool):
    """A tool for searching the web using a search API."""
    
    name: str = "WebSearchTool"
    description: str = "Searches the web for information on a given topic"
    api_key: Optional[str] = None
    search_engine: str = "google"
    max_results: int = 5
    
    def __init__(self, api_key: Optional[str] = None, search_engine: str = "google", max_results: int = 5):
        """Initialize the web search tool.
        
        Args:
            api_key: API key for the search engine (optional)
            search_engine: Search engine to use ('google', 'bing', etc.)
            max_results: Maximum number of results to return
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY")
        self.search_engine = search_engine
        self.max_results = max_results
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Search the web for the given query.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary containing search results
        """
        if self.search_engine == "google" and not self.api_key:
            # For demo purposes, return mock results when no API key is provided
            return self._mock_search_results(query)
        
        try:
            # This is a placeholder for an actual API call
            # In a real implementation, you would call the appropriate search API
            if self.search_engine == "google":
                return self._google_search(query)
            else:
                return {"error": f"Unsupported search engine: {self.search_engine}"}
                
        except Exception as e:
            return {"error": f"Error searching the web: {str(e)}"}
    
    def _mock_search_results(self, query: str) -> Dict[str, Any]:
        """Generate mock search results when no API key is available."""
        return {
            "query": query,
            "results": [
                {
                    "title": f"Search result 1 for {query}",
                    "link": f"https://example.com/result1?q={query}",
                    "snippet": f"This is a mock search result for {query}. In a real implementation, this would contain actual search results from the web.",
                },
                {
                    "title": f"Search result 2 for {query}",
                    "link": f"https://example.com/result2?q={query}",
                    "snippet": f"Another mock search result for {query}. Please provide a SEARCH_API_KEY environment variable to get real results.",
                }
            ],
            "note": "These are mock results. Add a SEARCH_API_KEY to your .env file to get real search results."
        }
    
    def _google_search(self, query: str) -> Dict[str, Any]:
        """Perform a Google search using the Custom Search API."""
        if not self.api_key:
            return self._mock_search_results(query)
            
        try:
            # This would be implemented with the actual Google Custom Search API
            # For now, return mock results
            return self._mock_search_results(query)
        except Exception as e:
            return {"error": f"Error with Google Search API: {str(e)}"}