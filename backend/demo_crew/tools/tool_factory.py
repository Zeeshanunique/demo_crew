"""
Tool factory that safely handles optional dependencies.
This allows the core app to run even if some tool dependencies aren't installed.
"""

from typing import List, Optional, Any, Dict, Callable
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class FallbackTool(BaseTool):
    """A fallback tool for when a required tool is not available due to missing dependencies."""
    
    name: str = "FallbackTool"
    description: str = "A fallback tool that reports missing dependencies"
    missing_dependency: str = "unknown"
    
    def __init__(self, name: str, description: str, missing_dependency: str):
        """Initialize the fallback tool.
        
        Args:
            name: Tool name
            description: Tool description
            missing_dependency: Name of the missing dependency
        """
        # Call super() with proper parameters for Pydantic validation
        super().__init__(name=name, description=description)
        self.missing_dependency = missing_dependency
    
    def _run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return an error message about the missing dependency.
        
        Returns:
            Dictionary with error message
        """
        return {
            "error": f"Missing dependency: {self.missing_dependency}",
            "message": f"The {self.name} is not available because the required dependency '{self.missing_dependency}' is not installed."
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
        from .image_tools import MistralOCRTool
        return MistralOCRTool()
    except ImportError:
        return FallbackTool(
            name="MistralOCRTool",
            description="Extracts text from images using Mistral",
            missing_dependency="mistral"
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
    """Get the Whisper transcription tool if available, or a fallback."""
    try:
        from .audio_tools import WhisperTranscriptionTool
        return WhisperTranscriptionTool(model_size="base")
    except ImportError:
        return FallbackTool(
            name="WhisperTranscriptionTool",
            description="Transcribes audio files",
            missing_dependency="whisper"
        )


def get_audio_analysis_tool() -> BaseTool:
    """Get the audio analysis tool if available, or a fallback."""
    try:
        from .audio_tools import AudioAnalysisTool
        return AudioAnalysisTool()
    except ImportError:
        return FallbackTool(
            name="AudioAnalysisTool",
            description="Analyzes audio files",
            missing_dependency="ffprobe"
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