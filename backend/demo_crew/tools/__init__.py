# Import custom tools with graceful fallback for missing dependencies
try:
    from .custom_tool import MyCustomTool as CustomTool
    HAS_CUSTOM_TOOL = True
except ImportError:
    HAS_CUSTOM_TOOL = False
    
try:
    from .audio_tools import WhisperTranscriptionTool
    HAS_WHISPER_TOOL = True
except ImportError:
    HAS_WHISPER_TOOL = False
    
try:
    from .video_tools import VideoPyDLPTool
    HAS_VIDEO_TOOL = True
except ImportError:
    HAS_VIDEO_TOOL = False
    
try:
    from .text_tools import OCRTool, WebSearchTool, DocumentRAGTool
    HAS_TEXT_TOOLS = True
except ImportError:
    HAS_TEXT_TOOLS = False
    
try:
    from .image_tools import ImageAnalysisTool
    HAS_IMAGE_TOOL = True
except ImportError:
    HAS_IMAGE_TOOL = False

# Define __all__ with only the tools that were successfully imported
__all__ = []

if HAS_CUSTOM_TOOL:
    __all__.append("CustomTool")
    
if HAS_WHISPER_TOOL:
    __all__.append("WhisperTranscriptionTool")
    
if HAS_VIDEO_TOOL:
    __all__.append("VideoPyDLPTool")
    
if HAS_TEXT_TOOLS:
    __all__.extend(["OCRTool", "WebSearchTool", "DocumentRAGTool"])
    
if HAS_IMAGE_TOOL:
    __all__.append("ImageAnalysisTool")
