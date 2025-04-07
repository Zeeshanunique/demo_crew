import os
from typing import Any, Dict
from langchain.tools import BaseTool
from faster_whisper import WhisperModel

class WhisperTranscriptionTool(BaseTool):
    """A tool for transcribing audio files using faster-whisper."""
    
    name: str = "WhisperTranscriptionTool"
    description: str = "Transcribes audio files using Whisper to extract spoken content"
    
    def __init__(self, model_size: str = "base"):
        """Initialize the Whisper transcription tool.
        
        Args:
            model_size: Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        super().__init__()
        # Using faster-whisper with CPU as the default device
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    def _run(self, audio_path: str) -> Dict[str, Any]:
        """Run the Whisper transcription on the provided audio file.
        
        Args:
            audio_path: Path to the audio file to transcribe
            
        Returns:
            Dictionary containing the transcription and segments with timestamps
        """
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found at path: {audio_path}"}
        
        try:
            # Transcribe the audio file with faster-whisper
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            
            # Process segments
            segments_list = []
            full_text = ""
            
            for segment in segments:
                segments_list.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": segment.avg_logprob
                })
                full_text += segment.text + " "
            
            return {
                "transcription": full_text.strip(),
                "segments": segments_list,
                "language": info.language
            }
        except Exception as e:
            return {"error": f"Error transcribing audio: {str(e)}"}