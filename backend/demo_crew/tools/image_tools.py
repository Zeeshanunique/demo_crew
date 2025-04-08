import os
import base64
import traceback
from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
from PIL import Image
import numpy as np
import cv2
import requests
from mistralai import Mistral
import dotenv
import json

# Try to import PaddleOCR as a fallback
try:
    from paddleocr import PaddleOCR
    HAS_PADDLE_OCR = True
except ImportError:
    HAS_PADDLE_OCR = False

# Load environment variables
dotenv.load_dotenv()

class MistralOCRTool(BaseTool):
    """Tool for performing OCR using Mistral AI's API."""
    
    name: str = "MistralOCRTool"
    description: str = "Extracts text from images using Mistral AI's document processing API"
    
    def _run(self, image_path: str = None, image_data: bytes = None):
        """
        Perform OCR on an image using Mistral AI.
        
        Args:
            image_path: Path to the image file
            image_data: Raw image data (alternative to image_path)
            
        Returns:
            Dictionary with OCR results
        """
        try:
            import requests
            import os
            from dotenv import load_dotenv
            
            # Load API key
            load_dotenv()
            api_key = os.environ.get("MISTRAL_API_KEY")
            
            if not api_key:
                # Try using OPENAI_API_KEY as fallback for demonstration
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    return {"error": "Missing MISTRAL_API_KEY environment variable"}
                print("Using OpenAI API key as fallback for Mistral OCR")
            
            # If image_data is provided directly, use it; otherwise read from file
            if image_data is None and image_path:
                if not os.path.exists(image_path):
                    return {"error": f"Image file not found: {image_path}"}
                with open(image_path, "rb") as f:
                    image_data = f.read()
            
            if image_data is None:
                return {"error": "No image data provided"}
            
            # Convert to base64 for API request
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            
            # Call Mistral AI OCR API (API url structure might vary based on documentation)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Following the documentation at https://docs.mistral.ai/capabilities/document/
            url = "https://api.mistral.ai/v1/document/analyze"
            
            payload = {
                "document": encoded_image,
                "document_type": "image",
                "task": "ocr"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                # Fall back to PaddleOCR if Mistral API fails
                return self._run_paddle_ocr(image_path, image_data)
            
            result = response.json()
            
            # Process the Mistral AI OCR response to match expected format
            extracted_text = ""
            text_blocks = []
            
            # Extract text from Mistral's response format
            # This structure needs to match Mistral's actual API response format
            if "pages" in result:
                for page in result["pages"]:
                    if "text" in page:
                        extracted_text += page["text"] + "\n\n"
                    if "blocks" in page:
                        text_blocks.extend(page["blocks"])
            elif "text" in result:
                # Direct text field
                extracted_text = result["text"]
            
            return {
                "extracted_text": extracted_text,
                "text_blocks": text_blocks,
                "raw_response": result
            }
            
        except Exception as e:
            print(f"Error with Mistral OCR: {str(e)}")
            # Fall back to PaddleOCR if exception occurs
            return self._run_paddle_ocr(image_path, image_data)
    
    def _run_paddle_ocr(self, image_path: str = None, image_data: bytes = None):
        """Fallback to PaddleOCR if Mistral API isn't available."""
        try:
            from paddleocr import PaddleOCR
            print("PaddleOCR initialized as fallback OCR engine.")
            
            # Create a temporary file if we have image_data
            if image_path is None and image_data is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(image_data)
                    image_path = temp_file.name
            
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
                
            # Initialize PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            
            # Perform OCR
            result = ocr.ocr(image_path, cls=True)
            
            # Process results
            extracted_text = ""
            text_blocks = []
            
            # Format PaddleOCR output
            if result and len(result) > 0:
                for idx, line in enumerate(result):
                    if isinstance(line, list) and len(line) > 0:
                        for text_item in line:
                            if len(text_item) == 2:
                                position, (text, confidence) = text_item
                                extracted_text += text + " "
                                text_blocks.append({
                                    "id": idx,
                                    "text": text,
                                    "bbox": position,
                                    "confidence": confidence
                                })
            
            return {
                "extracted_text": extracted_text.strip(),
                "text_blocks": text_blocks,
                "engine": "PaddleOCR"
            }
            
        except Exception as e:
            return {
                "error": f"OCR failed: {str(e)}",
                "traceback": traceback.format_exc()
            }

class ImageAnalysisTool(BaseTool):
    """Tool for analyzing images using computer vision techniques."""
    
    name: str = "ImageAnalysisTool"
    description: str = "Analyzes images for features, colors, and can extract text using OCR"
    
    def _run(
        self, 
        image_path: str, 
        extract_text: bool = True,
        analyze_colors: bool = True,
        detect_edges: bool = True,
        extract_features: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze an image using various computer vision techniques.
        
        Args:
            image_path: Path to the image file
            extract_text: Whether to extract text using OCR
            analyze_colors: Whether to analyze colors in the image
            detect_edges: Whether to detect edges in the image
            extract_features: Whether to extract features from the image
            
        Returns:
            Dictionary with analysis results
        """
        result = {}
        
        try:
            # Ensure the image file exists
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            # Extract text using OCR if requested
            if extract_text:
                try:
                    ocr_tool = MistralOCRTool()
                    ocr_result = ocr_tool._run(image_path)
                    result["text_extraction"] = {
                        "full_text": ocr_result.get("extracted_text", ""),
                        "text_blocks": ocr_result.get("text_blocks", [])
                    }
                except Exception as e:
                    result["text_extraction"] = {
                        "error": f"OCR failed: {str(e)}",
                        "full_text": ""
                    }
            
            # Analyze colors if requested
            if analyze_colors:
                try:
                    import cv2
                    import numpy as np
                    
                    # Read the image
                    image = cv2.imread(image_path)
                    
                    # Convert to RGB (OpenCV uses BGR)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Analyze colors
                    pixels = image_rgb.reshape((-1, 3))
                    from sklearn.cluster import KMeans
                    
                    # Use K-means to find the 5 most dominant colors
                    kmeans = KMeans(n_clusters=5)
                    kmeans.fit(pixels)
                    
                    # Get the RGB values of the cluster centers
                    colors = kmeans.cluster_centers_.astype(int)
                    
                    # Count pixels in each cluster to determine dominance
                    labels = kmeans.labels_
                    counts = np.bincount(labels)
                    
                    # Sort colors by frequency
                    color_info = []
                    for i in range(len(colors)):
                        color_info.append({
                            "rgb": colors[i].tolist(),
                            "hex": f"#{colors[i][0]:02x}{colors[i][1]:02x}{colors[i][2]:02x}",
                            "frequency": float(counts[i] / len(labels))
                        })
                    
                    # Sort by frequency
                    color_info = sorted(color_info, key=lambda x: x["frequency"], reverse=True)
                    
                    result["color_analysis"] = {
                        "dominant_colors": color_info,
                        "color_count": len(color_info)
                    }
                except Exception as e:
                    result["color_analysis"] = {
                        "error": f"Color analysis failed: {str(e)}"
                    }
            
            # Detect edges if requested
            if detect_edges:
                try:
                    import cv2
                    
                    # Read the image
                    image = cv2.imread(image_path)
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply Canny edge detection
                    edges = cv2.Canny(gray, 100, 200)
                    
                    # Count edge pixels
                    edge_count = int(np.sum(edges > 0))
                    
                    result["edge_detection"] = {
                        "edge_count": edge_count,
                        "edge_density": float(edge_count) / (edges.shape[0] * edges.shape[1])
                    }
                except Exception as e:
                    result["edge_detection"] = {
                        "error": f"Edge detection failed: {str(e)}"
                    }
            
            # Extract features if requested
            if extract_features:
                try:
                    import cv2
                    
                    # Read the image
                    image = cv2.imread(image_path)
                    
                    # Get image dimensions
                    height, width, channels = image.shape
                    
                    # Calculate aspect ratio
                    aspect_ratio = width / height
                    
                    # Calculate overall brightness
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    brightness = float(np.mean(gray))
                    
                    # Calculate contrast
                    contrast = float(np.std(gray))
                    
                    result["image_features"] = {
                        "dimensions": {
                            "width": width,
                            "height": height,
                            "aspect_ratio": aspect_ratio
                        },
                        "channels": channels,
                        "brightness": brightness,
                        "contrast": contrast,
                        "file_size_kb": os.path.getsize(image_path) / 1024
                    }
                except Exception as e:
                    result["image_features"] = {
                        "error": f"Feature extraction failed: {str(e)}"
                    }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Image analysis failed: {str(e)}",
                "traceback": traceback.format_exc()
            }