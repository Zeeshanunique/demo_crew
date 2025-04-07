#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime

from demo_crew.crew import DocumentCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.

def run():
    """
    Run the document processing crew.
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
        document_crew = DocumentCrew()
        crew = document_crew.create_crew()
        crew.kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'documents_dir': 'data/documents',
        'images_dir': 'data/images',
        'audio_dir': 'data/audio',
        'video_dir': 'data/video',
        'process_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        document_crew = DocumentCrew()
        crew = document_crew.create_crew()
        crew.train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        document_crew = DocumentCrew()
        crew = document_crew.create_crew()
        crew.replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'documents_dir': 'data/documents',
        'images_dir': 'data/images',
        'audio_dir': 'data/audio',
        'video_dir': 'data/video',
        'process_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        document_crew = DocumentCrew()
        crew = document_crew.create_crew()
        crew.test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
