import os
import re
import urllib.request
import json
from typing import Any, Dict, List, Optional
from crewai.tools import BaseTool
import pytesseract
from PIL import Image
import requests
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
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
    
    def __init__(self):
        """Initialize the OCR tool."""
        super().__init__()
    
    def _run(self, image_path: str, lang: str = "eng") -> Dict[str, Any]:
        """Extract text from the provided image.
        
        Args:
            image_path: Path to the image file
            lang: Language to use for OCR (default: 'eng')
            
        Returns:
            Dictionary containing extracted text
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found at path: {image_path}"}
        
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=lang)
            
            # Get more detailed data with bounding boxes
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Process and organize OCR results
            words = []
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    words.append({
                        'text': data['text'][i],
                        'conf': data['conf'][i],
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        },
                        'line_num': data['line_num'][i],
                        'block_num': data['block_num'][i]
                    })
            
            return {
                "full_text": text,
                "words": words,
                "language": lang
            }
        except Exception as e:
            return {"error": f"Error performing OCR: {str(e)}"}


class WebSearchTool(BaseTool):
    """A tool for performing web searches and retrieving information."""
    
    name: str = "WebSearchTool"
    description: str = "Searches the web for information on a given query"
    
    def __init__(self, max_results: int = 10, api_key: Optional[str] = None):
        """Initialize the web search tool.
        
        Args:
            max_results: Maximum number of search results to return
            api_key: Optional API key for search service
        """
        super().__init__()
        self.max_results = max_results
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY")
    
    def _run(self, query: str, num_results: Optional[int] = None) -> Dict[str, Any]:
        """Perform a web search for the given query.
        
        Args:
            query: Search query
            num_results: Number of results to return (overrides max_results)
            
        Returns:
            Dictionary containing search results
        """
        try:
            # For demo purposes, we're using a mock search result
            # In a real implementation, you would connect to a search API
            
            # Mock search for demonstration
            results = self._mock_search_results(query, num_results or self.max_results)
            
            return {
                "query": query,
                "results": results
            }
        except Exception as e:
            return {"error": f"Error performing web search: {str(e)}"}
    
    def _mock_search_results(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Generate mock search results for demonstration purposes.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries containing mock search results
        """
        # In a real implementation, this would call a search API
        results = []
        
        for i in range(min(num_results, 5)):
            results.append({
                "title": f"Search result {i+1} for: {query}",
                "link": f"https://example.com/result{i+1}",
                "snippet": f"This is a mock search result snippet for the query '{query}'. It contains some sample text that might be found in a real search result.",
                "source": "mock_search_engine"
            })
        
        return results


class DocumentRAGTool(BaseTool):
    """A tool for searching and retrieving information from documents using RAG."""
    
    name: str = "DocumentRAGTool"
    description: str = "Searches through documents to find relevant information using retrieval-augmented generation"
    
    def __init__(self, documents_dir: str = "knowledge"):
        """Initialize the document RAG tool.
        
        Args:
            documents_dir: Directory containing documents to index
        """
        super().__init__()
        self.documents_dir = documents_dir
        self.index = None
        self.embeddings = OpenAIEmbeddings()
        self.chunks = []
        self.source_documents = {}
    
    def _run(self, 
             query: str, 
             reload_index: bool = False, 
             k: int = 5) -> Dict[str, Any]:
        """Search for information related to the query in the indexed documents.
        
        Args:
            query: Search query
            reload_index: Whether to reload the document index
            k: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Load the index if it doesn't exist or reload is requested
            if self.index is None or reload_index:
                self._load_documents()
                self._create_index()
            
            # Search the index
            if not self.chunks:
                return {"error": "No documents have been indexed"}
            
            documents = self.index.similarity_search(query, k=k)
            
            # Format results
            results = []
            for doc in documents:
                source_file = self.source_documents.get(doc.metadata.get('source', ''), 'Unknown')
                results.append({
                    "content": doc.page_content,
                    "source": source_file,
                    "page": doc.metadata.get('page', None),
                    "metadata": doc.metadata
                })
            
            return {
                "query": query,
                "results": results
            }
        except Exception as e:
            return {"error": f"Error searching documents: {str(e)}"}
    
    def _load_documents(self) -> None:
        """Load documents from the documents directory."""
        self.chunks = []
        self.source_documents = {}
        
        if not os.path.exists(self.documents_dir):
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        for root, _, files in os.walk(self.documents_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Load document based on file extension
                    if file.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        documents = loader.load()
                    elif file.endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                        documents = loader.load()
                    elif file.endswith('.txt'):
                        loader = TextLoader(file_path)
                        documents = loader.load()
                    else:
                        continue
                    
                    # Split documents into chunks
                    chunks = text_splitter.split_documents(documents)
                    self.chunks.extend(chunks)
                    self.source_documents[file_path] = file
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    def _create_index(self) -> None:
        """Create a vector index from the document chunks."""
        if not self.chunks:
            return
        
        self.index = FAISS.from_documents(self.chunks, self.embeddings) 