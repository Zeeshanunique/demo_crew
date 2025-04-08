#!/usr/bin/env python
import sys
import warnings
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from demo_crew.document_graph import DocumentGraph
from demo_crew.master_agent import MasterAgent

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

def ensure_data_directories():
    """Create data directories if they don't exist"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories for different types of files
    for subdir in ["images", "audio", "video", "documents"]:
        os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
    
    return data_dir

def run_workflow():
    """
    Run the document processing workflow using LangGraph.
    """
    # Load environment variables from .env file
    load_environment()
    
    # Verify API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in a .env file or export it directly.")

    # Create data directories
    data_dir = ensure_data_directories()
    
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

def process_user_query(query: str) -> Dict[str, Any]:
    """
    Process a user query using the master agent
    
    Args:
        query: User's natural language query
        
    Returns:
        Dictionary with the result of processing the query
    """
    # Load environment variables from .env file
    load_environment()
    
    # Verify API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in a .env file or export it directly.")
    
    # Create data directories
    ensure_data_directories()
    
    try:
        # Initialize the master agent and process the query
        master_agent = MasterAgent()
        return master_agent.run(query)
    except Exception as e:
        raise Exception(f"An error occurred while processing the query: {e}")

def manipulate_dataset_schema(dataset_path: str, schema_changes: list, output_path: str = None) -> Dict[str, Any]:
    """
    Manipulate the schema of a dataset
    
    Args:
        dataset_path: Path to the JSON dataset file
        schema_changes: List of schema changes to apply
        output_path: Path to save the modified dataset (optional)
        
    Returns:
        Dictionary with the result of the operation
    """
    # Load environment variables from .env file
    load_environment()
    
    # Verify API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in a .env file or export it directly.")
    
    try:
        # Initialize the master agent
        master_agent = MasterAgent()
        
        # Use the schema manipulation tool directly
        schema_tool = next((tool for tool in master_agent.tools if tool.name == "SchemaManipulationTool"), None)
        if schema_tool is None:
            raise ValueError("Schema manipulation tool not found")
        
        return schema_tool._run(dataset_path, schema_changes, output_path)
    except Exception as e:
        raise Exception(f"An error occurred while manipulating the dataset: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Document Intelligence System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run the document processing workflow")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Process a natural language query")
    query_parser.add_argument("text", help="The query text")
    
    # Schema manipulation command
    schema_parser = subparsers.add_parser("schema", help="Manipulate a dataset schema")
    schema_parser.add_argument("dataset", help="Path to the dataset file")
    schema_parser.add_argument("changes_file", help="Path to a JSON file with schema changes")
    schema_parser.add_argument("--output", help="Output file path (optional)")
    
    return parser.parse_args()

def main():
    """Main entry point for command line interface"""
    args = parse_arguments()
    
    if args.command == "workflow":
        # Run the document processing workflow
        run_workflow()
    
    elif args.command == "query":
        # Process a query
        result = process_user_query(args.text)
        print(json.dumps(result, indent=2))
    
    elif args.command == "schema":
        # Load schema changes from file
        with open(args.changes_file, 'r') as f:
            changes = json.load(f)
        
        # Manipulate the dataset schema
        result = manipulate_dataset_schema(args.dataset, changes, args.output)
        print(json.dumps(result, indent=2))
    
    else:
        # Default to running the workflow
        run_workflow()

if __name__ == "__main__":
    main()
