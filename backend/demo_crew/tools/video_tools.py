import os
from typing import Any, Dict, List
from langchain.tools import BaseTool

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import tempfile
    import subprocess
    HAS_SUBPROCESS = True
except ImportError:
    HAS_SUBPROCESS = False

class VideoPyDLPTool(BaseTool):
    """A tool for processing video files with OpenCV and other utilities."""
    
    name: str = "VideoPyDLPTool"
    description: str = "Processes video files for frame extraction, audio transcription, and metadata analysis"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, video_path: str, extract_frames: bool = False, extract_audio: bool = False) -> Dict[str, Any]:
        """Process a video file.
        
        Args:
            video_path: Path to the video file to process
            extract_frames: Whether to extract key frames from the video
            extract_audio: Whether to extract and transcribe audio from the video
            
        Returns:
            Dictionary containing the video analysis results
        """
        # Check dependencies
        if not HAS_CV2:
            return {"error": "OpenCV (cv2) is not installed or missing system dependencies. Install with 'pip install opencv-python' and ensure system libraries are available."}
        
        if not os.path.exists(video_path):
            return {"error": f"Video file not found at path: {video_path}"}
        
        try:
            result = {"metadata": {}}
            
            # Open video file with OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"Could not open video file: {video_path}"}
            
            # Extract basic metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            result["metadata"] = {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": duration,
                "format": os.path.splitext(video_path)[1]
            }
            
            # Extract frames if requested
            if extract_frames and HAS_CV2:
                frames = self._extract_key_frames(cap, max_frames=5)
                result["frames"] = frames
            
            # Extract and transcribe audio if requested
            if extract_audio:
                if not HAS_WHISPER or not HAS_SUBPROCESS:
                    result["audio"] = {"error": "Missing dependencies for audio extraction. Install with 'pip install faster-whisper'}"}
                else:
                    audio = self._extract_and_transcribe_audio(video_path)
                    result["audio"] = audio
            
            # Close the video file
            cap.release()
            
            return result
            
        except Exception as e:
            return {"error": f"Error processing video: {str(e)}"}
    
    def _extract_key_frames(self, cap, max_frames=5):
        """Extract key frames from the video."""
        if not HAS_CV2:
            return {"error": "OpenCV (cv2) is not installed or missing system dependencies"}
            
        # Implementation for frame extraction
        # ...existing implementation...
        
    def _extract_and_transcribe_audio(self, video_path):
        """Extract audio from video and transcribe it."""
        if not HAS_WHISPER or not HAS_SUBPROCESS:
            return {"error": "Missing dependencies for audio extraction"}
            
        # Implementation for audio extraction
        # ...existing implementation...