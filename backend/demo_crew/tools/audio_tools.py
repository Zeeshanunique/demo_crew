import os
import difflib
import re
from typing import Any, Dict, List, Optional, Tuple
from langchain.tools import BaseTool
from faster_whisper import WhisperModel

def calculate_transcription_accuracy(transcription: str, ground_truth: str) -> Tuple[float, Dict]:
    """
    Calculate the accuracy of audio transcription compared to ground truth.
    
    Args:
        transcription: Text from audio transcription
        ground_truth: Known correct transcription
        
    Returns:
        Tuple of (accuracy score, detailed metrics)
    """
    # Normalize texts for comparison (lowercase, remove extra whitespace and punctuation)
    def normalize_text(text):
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text.lower())
        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    norm_transcription = normalize_text(transcription)
    norm_ground = normalize_text(ground_truth)
    
    # Calculate sentence-level accuracy
    trans_sentences = [s.strip() for s in re.split(r'[.!?]\s*', norm_transcription) if s.strip()]
    ground_sentences = [s.strip() for s in re.split(r'[.!?]\s*', norm_ground) if s.strip()]
    
    # Calculate word-level accuracy
    trans_words = set(norm_transcription.split())
    ground_words = set(norm_ground.split())
    
    if len(ground_words) == 0:
        word_accuracy = 0.0
    else:
        common_words = trans_words.intersection(ground_words)
        word_accuracy = len(common_words) / len(ground_words)
    
    # Calculate character-level accuracy
    matcher = difflib.SequenceMatcher(None, norm_transcription, norm_ground)
    char_accuracy = matcher.ratio()
    
    # Calculate weighted overall accuracy
    overall_accuracy = (char_accuracy * 0.3) + (word_accuracy * 0.7)
    
    # Detailed metrics
    metrics = {
        "character_match_ratio": char_accuracy,
        "word_match_ratio": word_accuracy,
        "overall_accuracy": overall_accuracy,
        "transcription_length": len(norm_transcription),
        "ground_truth_length": len(norm_ground),
        "confidence_level": get_confidence_level(overall_accuracy)
    }
    
    return overall_accuracy, metrics

def get_confidence_level(accuracy_score: float) -> str:
    """Convert accuracy score to confidence level string"""
    if accuracy_score >= 0.9:
        return "Very High"
    elif accuracy_score >= 0.7:
        return "High"
    elif accuracy_score >= 0.5:
        return "Moderate"
    elif accuracy_score >= 0.3:
        return "Low"
    else:
        return "Very Low"

class WhisperTranscriptionTool(BaseTool):
    """A tool for transcribing audio files using faster-whisper."""
    
    name: str = "WhisperTranscriptionTool"
    description: str = "Transcribes audio files using Whisper to extract spoken content"
    model: Any = None  # Adding the model field declaration so Pydantic won't raise an error
    
    def __init__(self, model_size: str = "base"):
        """Initialize the Whisper transcription tool.
        
        Args:
            model_size: Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        super().__init__()
        # Using faster-whisper with CPU as the default device
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    def _run(self, audio_path: str, ground_truth: str = None) -> Dict[str, Any]:
        """Run the Whisper transcription on the provided audio file.
        
        Args:
            audio_path: Path to the audio file to transcribe
            ground_truth: Optional ground truth text to calculate accuracy
            
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
            
            result = {
                "transcription": full_text.strip(),
                "segments": segments_list,
                "language": info.language
            }
            
            # Calculate accuracy if ground truth is provided
            if ground_truth:
                accuracy, accuracy_metrics = calculate_transcription_accuracy(
                    full_text.strip(), 
                    ground_truth
                )
                
                result["accuracy"] = accuracy_metrics
                
                # Print accuracy to terminal
                print(f"\n===== AUDIO TRANSCRIPTION ACCURACY =====")
                print(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']:.2%}")
                print(f"Character Match: {accuracy_metrics['character_match_ratio']:.2%}")
                print(f"Word Match: {accuracy_metrics['word_match_ratio']:.2%}")
                print(f"Confidence Level: {accuracy_metrics['confidence_level']}")
                print("=================================\n")
            
            return result
            
        except Exception as e:
            return {"error": f"Error transcribing audio: {str(e)}"}

# For image OCR
image_ground_truth = "This is a Iot of 12 point text to test the ocr code and see if it works on all types of file format.. The quick brown dog jumped over the lazy fox. The quick brown dog jumped. over the lazy fox. The quick brown dog jumped over the lazy fox. The quick brown dog jumped over the lazy fox."

# For audio transcription
audio_ground_truth = """1. "The stale smell of old beer lingers."
2. "It takes heat to bring out the odor."
3. "A cold dip restores health and zest."
4. "A salt pickle tastes fine with ham."
5. "Tacos al pastor are my favorite."
6. "A zestful food is the hot cross bun."

Language: English"""