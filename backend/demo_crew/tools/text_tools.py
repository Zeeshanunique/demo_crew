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
    
    def _run(self, query: str, top_k: int = 5, single_file: str = None) -> Dict[str, Any]:
        """Search for information in the indexed documents or process a single file.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            single_file: Optional path to a single file to process instead of the index
            
        Returns:
            Dictionary containing search results or processed file content
        """
        # If a single file is provided, process it directly instead of using the index
        if single_file and os.path.exists(single_file):
            return self._process_single_file(single_file)
            
        # Ensure index is built for regular searches
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
    
    def _process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single document file and extract its content.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary containing the extracted content
        """
        if not os.path.exists(file_path):
            return {'error': f"File not found: {file_path}"}
            
        try:
            # Determine the file type based on extension
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Load document based on file extension
            if file_extension == '.pdf':
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                except Exception as pdf_error:
                    # If PyPDFLoader fails, attempt to use a different PDF loader
                    from langchain_community.document_loaders import UnstructuredPDFLoader
                    loader = UnstructuredPDFLoader(file_path)
                    docs = loader.load()
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
                docs = loader.load()
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            else:
                return {'error': f"Unsupported file type: {file_extension}"}
            
            # Combine content from all pages
            all_content = ""
            metadata = {}
            
            for doc in docs:
                all_content += doc.page_content + "\n\n"
                # Merge metadata
                metadata.update(doc.metadata)
            
            # Extract main elements from the content
            structured_data = self._extract_structured_data(all_content)
            
            return {
                'success': True,
                'filename': os.path.basename(file_path),
                'content': all_content.strip(),
                'metadata': metadata,
                'structured_data': structured_data,
                'page_count': len(docs)
            }
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            return {
                'error': f"Error processing file: {str(e)}",
                'traceback': trace,
                'file_path': file_path
            }
    
    def _extract_structured_data(self, content: str) -> Dict[str, Any]:
        """Extract structured data from document content.
        
        Args:
            content: The document content as text
            
        Returns:
            Dictionary containing extracted structured data
        """
        # This is a simple implementation that can be expanded with more sophisticated extraction
        structured_data = {
            'summary': content[:500] + "..." if len(content) > 500 else content,
            'detected_entities': {}
        }
        
        # Simple entity extraction for common patterns
        # Dates (simple pattern)
        dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}', content)
        if dates:
            structured_data['detected_entities']['dates'] = dates[:10]  # Limit to 10 dates
            
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        if emails:
            structured_data['detected_entities']['emails'] = emails[:10]
            
        # Monetary values
        monetary = re.findall(r'\$\s?\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s?(?:dollars|USD|EUR|GBP|JPY)', content)
        if monetary:
            structured_data['detected_entities']['monetary_values'] = monetary[:10]
            
        # Phone numbers (simple pattern)
        phones = re.findall(r'(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}', content)
        if phones:
            structured_data['detected_entities']['phone_numbers'] = phones[:10]
            
        return structured_data
    
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