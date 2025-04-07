"""
Tool factory that safely handles optional dependencies.
This allows the core app to run even if some tool dependencies aren't installed.
"""

from typing import List, Optional, Any, Dict, Callable
from langchain.tools import BaseTool


class FallbackTool(BaseTool):
    """A tool that provides a fallback when the required dependencies aren't installed."""
    
    def __init__(self, name: str, description: str, missing_dependency: str):
        """Initialize the fallback tool.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            missing_dependency: Name of the missing dependency
        """
        self.name = name
        self.description = description
        self.missing_dependency = missing_dependency
        super().__init__()
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Provide information about the missing dependency."""
        return {
            "error": f"Tool '{self.name}' cannot be used because dependency '{self.missing_dependency}' is not installed.",
            "solution": f"Install the optional dependencies with: pip install demo_crew[tools]"
        }


def get_document_rag_tool() -> BaseTool:
    """Get the document RAG tool if available, or a fallback."""
    try:
        from .text_tools import DocumentRAGTool
        return DocumentRAGTool()
    except ImportError:
        return FallbackTool(
            name="DocumentRAGTool",
            description="Searches through documents to find relevant information",
            missing_dependency="langchain-community"
        )


def get_web_search_tool() -> BaseTool:
    """Get the web search tool if available, or a fallback."""
    try:
        from .text_tools import WebSearchTool
        return WebSearchTool()
    except ImportError:
        return FallbackTool(
            name="WebSearchTool",
            description="Searches the web for information",
            missing_dependency="requests"
        )


def get_ocr_tool() -> BaseTool:
    """Get the OCR tool if available, or a fallback."""
    try:
        from .text_tools import OCRTool
        return OCRTool()
    except ImportError:
        return FallbackTool(
            name="OCRTool",
            description="Extracts text from images",
            missing_dependency="pytesseract"
        )


def get_image_analysis_tool() -> BaseTool:
    """Get the image analysis tool if available, or a fallback."""
    try:
        from .image_tools import ImageAnalysisTool
        return ImageAnalysisTool()
    except ImportError:
        return FallbackTool(
            name="ImageAnalysisTool",
            description="Analyzes images for features and content",
            missing_dependency="opencv-python"
        )


def get_whisper_transcription_tool() -> BaseTool:
    """Get the whisper transcription tool if available, or a fallback."""
    try:
        from .audio_tools import WhisperTranscriptionTool
        return WhisperTranscriptionTool()
    except ImportError:
        return FallbackTool(
            name="WhisperTranscriptionTool",
            description="Transcribes audio files",
            missing_dependency="openai-whisper"
        )


def get_video_tool() -> BaseTool:
    """Get the video processing tool if available, or a fallback."""
    try:
        from .video_tools import VideoPyDLPTool
        return VideoPyDLPTool()
    except ImportError:
        return FallbackTool(
            name="VideoPyDLPTool",
            description="Processes video files",
            missing_dependency="opencv-python and openai-whisper"
        )