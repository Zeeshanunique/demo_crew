from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import uuid
import time
import shutil
import json
from dotenv import load_dotenv
import asyncio
import sys

# Add parent directory to path to ensure imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from demo_crew.master_agent import MasterAgent

# Create router
router = APIRouter(tags=["Upload"])

# Create uploads directory structure if it doesn't exist
def ensure_upload_dirs():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "uploads")
    for folder in ["text", "image", "video", "audio", "processed"]:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)
    return base_dir

# Function to process files in the background 
async def process_file(file_path: str, file_type: str, instructions: Optional[str] = None):
    """
    Process a file based on its type using the appropriate AI agent.
    """
    start_time = time.time()
    
    try:
        # Load environment variables for OpenAI API key
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env")
        load_dotenv(dotenv_path)
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in a .env file.")
        
        # Create a specific query based on the file type and instructions
        query = f"Process this {file_type} file"
        if instructions:
            query += f". Instructions: {instructions}"
        
        # Initialize and use document graph directly for file processing
        from demo_crew.document_graph import DocumentGraph
        document_graph = DocumentGraph()
        
        # Process the file using the document graph with direct file path
        processing_result = document_graph.run(
            inputs={},  # Empty inputs, we'll specify the file directly
            direct_file_path=file_path,
            file_type=file_type
        )
        
        # Extract the relevant result based on file type
        output_key = f"{file_type}_processing_result"
        output = processing_result.get(output_key, "No output generated")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result format
        result = {
            "status": "completed",
            "file_type": file_type,
            "original_file": os.path.basename(file_path),
            "processing_time": f"{processing_time:.2f} seconds",
            "instructions": instructions if instructions else "No specific instructions provided",
            "extracted_data": {
                "output": output,
                "agent_type": file_type,
                "timestamp": time.time(),
                "processed_file": file_path
            }
        }
        
    except Exception as e:
        import traceback
        # Handle any exceptions during processing
        result = {
            "status": "failed",
            "file_type": file_type,
            "original_file": os.path.basename(file_path),
            "processing_time": f"{time.time() - start_time:.2f} seconds",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "instructions": instructions if instructions else "No specific instructions provided",
        }
    
    # Save result to a JSON file
    result_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_result.json"
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "processed")
    result_path = os.path.join(processed_dir, result_filename)
    
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    
    return result_path

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_type: str = Form(...),
    instructions: Optional[str] = Form(None)
):
    """
    Upload a file for processing by AI agents
    """
    base_dir = ensure_upload_dirs()
    
    # Validate file type
    if file_type not in ["text", "image", "video", "audio"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Generate a unique filename
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1] if original_filename else ""
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Save the file
    file_path = os.path.join(base_dir, file_type, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file in the background
    background_tasks.add_task(process_file, file_path, file_type, instructions)
    
    return {
        "status": "processing",
        "message": "File uploaded and being processed",
        "file_id": unique_filename,
        "original_filename": original_filename,
        "file_type": file_type
    }

@router.get("/status/{file_id}")
async def get_processing_status(file_id: str):
    """
    Check the status of a file being processed
    """
    # In a real implementation, this would check a database or queue
    # For demonstration, we'll simulate completed status
    
    base_dir = ensure_upload_dirs()
    processed_dir = os.path.join(base_dir, "processed")
    
    # Check if result file exists (without extension)
    file_id_base = os.path.splitext(file_id)[0]
    result_files = [f for f in os.listdir(processed_dir) if f.startswith(file_id_base)]
    
    if result_files:
        result_file = result_files[0]
        result_path = os.path.join(processed_dir, result_file)
        
        with open(result_path, "r") as f:
            result_data = json.load(f)
        
        return {
            "status": "completed",
            "result": result_data,
            "result_file": f"/uploads/processed/{result_file}"
        }
    
    return {
        "status": "processing",
        "message": "File is still being processed"
    }