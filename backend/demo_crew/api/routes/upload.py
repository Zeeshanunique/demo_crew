from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import uuid
import time
import shutil
import json

# Create router
router = APIRouter(tags=["Upload"])

# Create uploads directory structure if it doesn't exist
def ensure_upload_dirs():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "uploads")
    for folder in ["text", "image", "video", "audio", "processed"]:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)
    return base_dir

# Function to process files in the background (simulated for now)
async def process_file(file_path: str, file_type: str, instructions: Optional[str] = None):
    """
    Process a file based on its type. This would invoke the appropriate AI agents.
    For demonstration, we'll simulate processing time and return a simple result.
    """
    # In a real implementation, this would dispatch to the appropriate CrewAI agent
    # based on the file type and instructions
    
    # Simulate processing time
    time.sleep(2)
    
    # Generate a result JSON
    result = {
        "status": "completed",
        "file_type": file_type,
        "original_file": os.path.basename(file_path),
        "processing_time": "2 seconds",
        "instructions": instructions if instructions else "No specific instructions provided",
        "extracted_data": {
            "sample_key": "This is sample extracted data",
            "confidence": 0.95,
            "timestamp": time.time()
        }
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