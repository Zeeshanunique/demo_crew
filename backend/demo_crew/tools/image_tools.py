import os
from typing import Any, Dict, List, Optional
from crewai.tools import BaseTool
from PIL import Image
import numpy as np
import cv2

class ImageAnalysisTool(BaseTool):
    """A tool for analyzing images to extract visual information and features."""
    
    name: str = "ImageAnalysisTool"
    description: str = "Analyzes images to extract features, objects, colors, and other visual information"
    
    def __init__(self):
        """Initialize the image analysis tool."""
        super().__init__()
    
    def _run(self, 
             image_path: str, 
             analyze_colors: bool = True,
             detect_edges: bool = True,
             extract_features: bool = True) -> Dict[str, Any]:
        """Analyze the provided image.
        
        Args:
            image_path: Path to the image file
            analyze_colors: Whether to analyze color distribution
            detect_edges: Whether to detect edges in the image
            extract_features: Whether to extract key features
            
        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found at path: {image_path}"}
        
        try:
            # Open the image
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            
            # Convert to BGR for OpenCV processing
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            result = {
                "image_path": image_path,
                "dimensions": {
                    "width": pil_image.width,
                    "height": pil_image.height
                },
                "format": pil_image.format,
                "mode": pil_image.mode
            }
            
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
            return {"error": f"Error analyzing image: {str(e)}"}
    
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