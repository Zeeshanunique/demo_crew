import os
from typing import Any, Dict, List, Optional
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

# Add MoviePy dependency check
try:
    import moviepy.editor as mp
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

class VideoPyDLPTool(BaseTool):
    """A tool for processing video files with OpenCV and other utilities."""
    
    name: str = "VideoPyDLPTool"
    description: str = "Processes video files for frame extraction, audio transcription, and metadata analysis"
    
    # Define whisper_model as an optional field to comply with Pydantic requirements
    whisper_model: Optional[Any] = None
    
    def __init__(self):
        super().__init__()
        # Initialize Whisper model if available
        if HAS_WHISPER:
            try:
                self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            except Exception as e:
                print(f"Warning: Could not initialize Whisper model: {str(e)}")
    
    def _run(self, video_path: str, extract_frames: bool = False, extract_audio: bool = True) -> Dict[str, Any]:
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
                if not HAS_WHISPER:
                    result["audio"] = {"error": "Missing dependency: faster-whisper. Install with 'pip install faster-whisper'"}
                elif not (HAS_MOVIEPY or HAS_SUBPROCESS):
                    result["audio"] = {"error": "Missing dependencies for audio extraction. Install with 'pip install moviepy' or ensure ffmpeg is available."}
                else:
                    audio = self._extract_and_transcribe_audio(video_path)
                    result["audio"] = audio
            
            # Close the video file
            cap.release()
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "error": f"Error processing video: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _extract_key_frames(self, cap, max_frames=5):
        """Extract key frames from the video."""
        if not HAS_CV2:
            return {"error": "OpenCV (cv2) is not installed or missing system dependencies"}
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            return {"error": "Video has no frames"}
        
        # For simple implementation, extract evenly spaced frames
        frame_interval = total_frames // min(max_frames, total_frames)
        if frame_interval <= 0:
            frame_interval = 1
        
        frame_positions = []
        for i in range(min(max_frames, total_frames)):
            frame_pos = i * frame_interval
            if frame_pos < total_frames:
                frame_positions.append(frame_pos)
        
        # Extract frames at the calculated positions
        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            success, frame = cap.read()
            if success:
                # Convert to jpg for easier handling
                _, buffer = cv2.imencode('.jpg', frame)
                jpg_as_text = buffer.tobytes()
                
                # Get frame timestamp
                timestamp = pos / cap.get(cv2.CAP_PROP_FPS)
                
                frames.append({
                    "frame_number": pos,
                    "timestamp": timestamp,
                    "image_data_length": len(jpg_as_text),
                    "image_format": "jpg"
                })
        
        return frames
    
    def _extract_audio_with_moviepy(self, video_path):
        """Extract audio from video using MoviePy."""
        if not HAS_MOVIEPY:
            return None, "MoviePy is not installed"
        
        try:
            # Create a temporary file for the extracted audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Extract audio using MoviePy
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio_path, logger=None)
            video.close()
            
            # Check if the audio file was created and has content
            if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return None, "No audio track found in video or extraction failed"
            
            return temp_audio_path, None
            
        except Exception as e:
            # Clean up any temporary file that might have been created
            try:
                if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except:
                pass
            
            return None, f"Error extracting audio with MoviePy: {str(e)}"
    
    def _extract_audio_with_ffmpeg(self, video_path):
        """Extract audio from video using ffmpeg subprocess."""
        if not HAS_SUBPROCESS:
            return None, "Subprocess module is not available"
        
        try:
            # Create a temporary file for the extracted audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Use ffmpeg to extract audio from the video
            cmd = [
                'ffmpeg', 
                '-i', video_path,
                '-q:a', '0',
                '-map', 'a',
                '-f', 'wav',
                temp_audio_path
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            _, stderr = process.communicate()
            
            if process.returncode != 0:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return None, f"Failed to extract audio: {stderr.decode('utf-8', errors='replace')}"
            
            # Check if the audio file was created and has content
            if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return None, "No audio track found in video or extraction failed"
            
            return temp_audio_path, None
            
        except Exception as e:
            # Clean up any temporary file that might have been created
            try:
                if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except:
                pass
            
            return None, f"Error extracting audio with ffmpeg: {str(e)}"
    
    def _extract_and_transcribe_audio(self, video_path):
        """Extract audio from video and transcribe it."""
        if not HAS_WHISPER:
            return {"error": "Missing dependency: faster-whisper"}
        
        try:
            # Try MoviePy first if available (more reliable for many file formats)
            if HAS_MOVIEPY:
                temp_audio_path, error = self._extract_audio_with_moviepy(video_path)
                if error and HAS_SUBPROCESS:
                    # Fall back to ffmpeg if MoviePy fails
                    print(f"MoviePy audio extraction failed: {error}. Trying ffmpeg...")
                    temp_audio_path, error = self._extract_audio_with_ffmpeg(video_path)
            elif HAS_SUBPROCESS:
                # Use ffmpeg if MoviePy is not available
                temp_audio_path, error = self._extract_audio_with_ffmpeg(video_path)
            else:
                return {"error": "No audio extraction method available. Install MoviePy or ensure ffmpeg is available."}
            
            # Check if extraction was successful
            if error:
                return {"error": error}
            
            # Initialize Whisper model if it's not already initialized
            if self.whisper_model is None and HAS_WHISPER:
                self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            
            # Transcribe the audio
            segments, info = self.whisper_model.transcribe(temp_audio_path, beam_size=5)
            
            # Process segments to create a readable transcript
            segments_data = []
            full_transcript = ""
            
            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": float(segment.avg_logprob)
                }
                segments_data.append(segment_data)
                full_transcript += segment.text + " "
            
            # Clean up the temporary file
            os.unlink(temp_audio_path)
            
            return {
                "transcript": full_transcript.strip(),
                "segments": segments_data,
                "language": info.language,
                "audio_duration": info.duration
            }
            
        except Exception as e:
            import traceback
            return {
                "error": f"Error during audio processing: {str(e)}",
                "traceback": traceback.format_exc()
            }