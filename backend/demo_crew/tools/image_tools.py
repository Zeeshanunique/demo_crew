import os
import base64
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

class ImageAnalysisTool(BaseTool):
    """A tool for analyzing images to extract visual information and features."""
    
    name: str = "ImageAnalysisTool"
    description: str = "Extracts text from images using OCR and analyzes image content"
    mistral_client: Any = None
    paddle_ocr: Any = None
    
    def __init__(self):
        """Initialize the image analysis tool with Mistral OCR capabilities and PaddleOCR fallback."""
        super().__init__()
        
        # Initialize Mistral client
        api_key = os.environ.get("MISTRAL_API_KEY")
        if api_key:
            self.mistral_client = Mistral(api_key=api_key)
        else:
            print("Warning: MISTRAL_API_KEY not found. OCR may fall back to PaddleOCR or not work.")
            
        # Initialize PaddleOCR as fallback
        if HAS_PADDLE_OCR:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                print("PaddleOCR initialized as fallback OCR engine.")
            except Exception as e:
                print(f"Warning: Could not initialize PaddleOCR: {str(e)}")
    
    def _run(self, 
             image_path: str, 
             analyze_colors: bool = False,
             detect_edges: bool = False,
             extract_features: bool = False,
             extract_text: bool = True,
             lang: str = "en") -> Dict[str, Any]:
        """Analyze the provided image with emphasis on text extraction.
        
        Args:
            image_path: Path to the image file
            analyze_colors: Whether to analyze color distribution
            detect_edges: Whether to detect edges in the image
            extract_features: Whether to extract key features
            extract_text: Whether to extract text using OCR (default: True)
            lang: Language for OCR processing
            
        Returns:
            Dictionary containing text extraction results and optional image analysis
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found at path: {image_path}"}
        
        try:
            result = {"image_path": image_path}
            
            # Extract text using OCR if requested (default is True)
            if extract_text:
                if self.mistral_client:
                    ocr_results = self._extract_text_mistral(image_path)
                    result["text_extraction"] = ocr_results
                elif self.paddle_ocr:
                    ocr_results = self._extract_text_paddle(image_path)
                    result["text_extraction"] = ocr_results
                else:
                    return {
                        "error": "No OCR engine available. Cannot perform OCR.",
                        "image_path": image_path
                    }
            
            # Only perform these analyses if specifically requested
            # Open the image for additional analyses if needed
            if analyze_colors or detect_edges or extract_features:
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                
                # Convert to BGR for OpenCV processing if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image
                
                result["dimensions"] = {
                    "width": pil_image.width,
                    "height": pil_image.height
                }
                result["format"] = pil_image.format
                result["mode"] = pil_image.mode
                
                # Analyze colors if requested
                if analyze_colors:
                    result["colors"] = self._analyze_colors(pil_image)
                
                # Detect edges if requested
                if detect_edges:
                    result["edges"] = self._detect_edges(image_bgr, image_path)
                
                # Extract features if requested
                if extract_features:
                    result["features"] = self._extract_features(image_bgr)
            
            return result
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def _extract_text_mistral(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using Mistral OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text
        """
        try:
            # Read the image file and encode it as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Process the image with Mistral OCR API
            ocr_response = self.mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}" 
                }
            )
            
            # Extract the text from the OCR response
            if hasattr(ocr_response, 'text'):
                extracted_text = ocr_response.text
            else:
                # If the response structure is different, try to access it as a dictionary
                extracted_text = ocr_response.get('text', '')
                
            # Create a simplified response structure
            return {
                "full_text": extracted_text,
                "text_regions": [{"text": extracted_text}],
                "num_regions": 1,
                "provider": "mistral-ocr"
            }
        except Exception as e:
            print(f"Error with Mistral OCR: {str(e)}.")
            return {
                "error": f"Error extracting text with Mistral: {str(e)}",
                "full_text": "",
                "text_regions": []
            }
    
    def _extract_text_paddle(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using PaddleOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text
        """
        try:
            # Read the image file
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # Process the image with PaddleOCR
            ocr_results = self.paddle_ocr.ocr(image_np, cls=True)
            
            # Extract the text from the OCR response
            extracted_text = " ".join([line[1][0] for line in ocr_results[0]])
            
            # Create a simplified response structure
            return {
                "full_text": extracted_text,
                "text_regions": [{"text": extracted_text}],
                "num_regions": len(ocr_results[0]),
                "provider": "paddle-ocr"
            }
        except Exception as e:
            print(f"Error with PaddleOCR: {str(e)}.")
            return {
                "error": f"Error extracting text with PaddleOCR: {str(e)}",
                "full_text": "",
                "text_regions": []
            }
    
    def _analyze_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze color distribution in the image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary containing color analysis
        """
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image for faster processing
        image_resized = image.resize((100, 100))
        pixels = list(image_resized.getdata())
        
        # Calculate average color
        avg_r = sum(p[0] for p in pixels) // len(pixels)
        avg_g = sum(p[1] for p in pixels) // len(pixels)
        avg_b = sum(p[2] for p in pixels) // len(pixels)
        
        # Dominant colors
        color_counts = {}
        for pixel in pixels:
            # Simplify color by reducing precision
            simple_pixel = (pixel[0] // 32 * 32, pixel[1] // 32 * 32, pixel[2] // 32 * 32)
            color_counts[simple_pixel] = color_counts.get(simple_pixel, 0) + 1
        
        # Get the most common colors
        dominant_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "average_color": {
                "r": avg_r,
                "g": avg_g,
                "b": avg_b,
                "hex": f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}"
            },
            "dominant_colors": [
                {
                    "r": color[0][0],
                    "g": color[0][1],
                    "b": color[0][2],
                    "hex": f"#{color[0][0]:02x}{color[0][1]:02x}{color[0][2]:02x}",
                    "percentage": (color[1] * 100) // len(pixels)
                }
                for color in dominant_colors
            ]
        }
    
    def _detect_edges(self, image: np.ndarray, image_path: str) -> Dict[str, Any]:
        """Detect edges in the image.
        
        Args:
            image: OpenCV image
            image_path: Path to save edge detection results
            
        Returns:
            Dictionary containing edge detection results
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        
        # Save edge detection result
        edge_path = os.path.splitext(image_path)[0] + "_edges.jpg"
        cv2.imwrite(edge_path, edges)
        
        return {
            "edge_density": edge_density,
            "edge_image_path": edge_path
        }
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract key features from the image.
        
        Args:
            image: OpenCV image
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Convert to grayscale for feature detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # SIFT feature detector
        try:
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            features["number_of_keypoints"] = len(keypoints)
            
            if len(keypoints) > 0:
                # Calculate distribution of keypoints
                heights, widths = gray.shape
                quadrants = {
                    "top_left": 0,
                    "top_right": 0,
                    "bottom_left": 0,
                    "bottom_right": 0
                }
                
                for kp in keypoints:
                    x, y = kp.pt
                    if x < widths/2 and y < heights/2:
                        quadrants["top_left"] += 1
                    elif x >= widths/2 and y < heights/2:
                        quadrants["top_right"] += 1
                    elif x < widths/2 and y >= heights/2:
                        quadrants["bottom_left"] += 1
                    else:
                        quadrants["bottom_right"] += 1
                
                features["keypoint_distribution"] = {
                    k: round(v / len(keypoints), 2) for k, v in quadrants.items()
                }
        except Exception as e:
            features["sift_error"] = str(e)
        
        # Basic structural analysis
        features["structural"] = {
            "aspect_ratio": image.shape[1] / image.shape[0] if image.shape[0] > 0 else 0,
            "resolution": image.shape[0] * image.shape[1]
        }
        
        return features