#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime

from demo_crew.document_graph import DocumentGraph

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# document processing workflow locally.

def run():
    """
    Run the document processing workflow using LangGraph.
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories for different types of files
    for subdir in ["images", "audio", "video", "documents"]:
        os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
    
    inputs = {
        'documents_dir': os.path.join(data_dir, "documents"),
        'images_dir': os.path.join(data_dir, "images"),
        'audio_dir': os.path.join(data_dir, "audio"),
        'video_dir': os.path.join(data_dir, "video"),
        'process_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'output_format': 'json'
    }
    
    try:
        # Initialize the document graph and run the workflow
        document_graph = DocumentGraph()
        result = document_graph.run(inputs)
        
        print("\n=== Document Processing Complete ===")
        print(f"Text Processing Result: {result.get('text_processing_result', 'N/A')[:100]}...")
        print(f"Image Processing Result: {result.get('image_processing_result', 'N/A')[:100]}...")
        print(f"Video Processing Result: {result.get('video_processing_result', 'N/A')[:100]}...")
        print(f"Audio Processing Result: {result.get('audio_processing_result', 'N/A')[:100]}...")
        print(f"Output saved to: processed_data.json")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the workflow: {e}")


if __name__ == "__main__":
    run()
