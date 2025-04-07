#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from demo_crew.document_graph import DocumentGraph

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def load_environment():
    """
    Load environment variables from various potential .env file locations
    """
    # Try to load from the backend directory first
    backend_env = Path(__file__).parent.parent.parent / ".env"
    if backend_env.exists():
        load_dotenv(backend_env)
        print(f"Loaded environment from {backend_env}")
        return

    # Try current working directory
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env)
        print(f"Loaded environment from {cwd_env}")
        return
    
    # Try parent directories
    parent_env = Path.cwd().parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env)
        print(f"Loaded environment from {parent_env}")
        return
    
    # Check if environment variables are already set
    if os.environ.get("OPENAI_API_KEY"):
        print("Environment variables already loaded")
        return
    
    print("Warning: No .env file found, using default environment variables")

def run():
    """
    Run the document processing workflow using LangGraph.
    """
    # Load environment variables from .env file
    load_environment()
    
    # Verify API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in a .env file or export it directly.")

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
