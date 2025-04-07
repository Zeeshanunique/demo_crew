import os
from typing import Any, Dict, List
from crewai.tools import BaseTool
import cv2
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import subprocess

class VideoPyDLPTool(BaseTool):
    """A tool for processing video files to extract frames, audio, and analyze content."""
    
    name: str = "VideoPyDLPTool"
    description: str = "Processes video files to extract frames, audio transcription, and key visual elements"
    
    def __init__(self, frame_interval: int = 30, whisper_model_size: str = "base"):
        """Initialize the video processing tool.
        
        Args:
            frame_interval: Number of frames to skip between extractions
            whisper_model_size: Size of the Whisper model for audio transcription
        """
        super().__init__()
        self.frame_interval = frame_interval
        # Using faster-whisper with CPU as the default device
        self.whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
    
    def _run(self, 
             video_path: str, 
             extract_frames: bool = True, 
             transcribe_audio: bool = True,
             output_dir: str = "output") -> Dict[str, Any]:
        """Process the video file to extract frames and transcribe audio.
        
        Args:
            video_path: Path to the video file
            extract_frames: Whether to extract frames from the video
            transcribe_audio: Whether to transcribe audio from the video
            output_dir: Directory to save extracted frames
            
        Returns:
            Dictionary containing extracted data from the video
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found at path: {video_path}"}
        
        result = {"video_path": video_path}
        
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Extract frames if requested
            if extract_frames:
                frames_info = self._extract_frames(video_path, output_dir)
                result["frames"] = frames_info
            
            # Transcribe audio if requested
            if transcribe_audio:
                transcription = self._transcribe_audio(video_path)
                result["audio"] = transcription
                
            return result
        except Exception as e:
            return {"error": f"Error processing video: {str(e)}"}
    
    def _extract_frames(self, video_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """Extract frames from the video at specified intervals.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            
        Returns:
            List of dictionaries containing frame information
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        frames_info = []
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % self.frame_interval == 0:
                timestamp = count / fps
                frame_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Basic frame analysis (can be expanded)
                frames_info.append({
                    "frame_number": count,
                    "timestamp": timestamp,
                    "path": frame_path,
                    "time_str": self._format_timestamp(timestamp)
                })
                
            count += 1
                
        cap.release()
        
        return frames_info
    
    def _transcribe_audio(self, video_path: str) -> Dict[str, Any]:
        """Extract and transcribe audio from the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing audio transcription
        """
        try:
            # Create a temporary file for audio extraction
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            # Extract audio using ffmpeg if available, otherwise return a placeholder
            try:
                # Try to extract audio using ffmpeg subprocess (more compatible)
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-q:a', '0',
                    '-map', 'a',
                    temp_audio_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Transcribe audio using faster-whisper
                segments, info = self.whisper_model.transcribe(temp_audio_path, beam_size=5)
                
                # Process segments
                segments_list = []
                full_text = ""
                
                for segment in segments:
                    segments_list.append({
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        "start_str": self._format_timestamp(segment.start),
                        "end_str": self._format_timestamp(segment.end),
                        "confidence": segment.avg_logprob
                    })
                    full_text += segment.text + " "
                
                result = {
                    "transcription": full_text.strip(),
                    "segments": segments_list,
                    "language": info.language
                }
                
            except (subprocess.SubprocessError, FileNotFoundError):
                # If ffmpeg isn't available or fails, return a placeholder
                result = {
                    "transcription": "Audio extraction requires ffmpeg to be installed. Using placeholder transcription.",
                    "segments": [
                        {
                            "text": "Audio transcription placeholder. Install ffmpeg for actual transcription.",
                            "start": 0,
                            "end": 1,
                            "start_str": "00:00:00.00",
                            "end_str": "00:00:01.00",
                            "confidence": 0
                        }
                    ],
                    "language": "en"
                }
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            return result
            
        except Exception as e:
            return {
                "transcription": f"Error transcribing audio: {str(e)}",
                "segments": [],
                "language": "unknown"
            }
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into a readable timestamp.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string (HH:MM:SS)
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"