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
        
        # Save result to a JSON file
        result_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_result.json"
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "processed")
        result_path = os.path.join(processed_dir, result_filename)
        
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        
        # NEW CODE: Automatically run OutputExtractorAgent to extract and format the data
        try:
            # Initialize the MasterAgent
            master_agent = MasterAgent()
            
            # Find the OutputExtractorAgent tool
            output_extractor = next((tool for tool in master_agent.tools if tool.name == "OutputExtractorAgent"), None)
            
            if output_extractor:
                # Get the paths for the datasets
                backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                combined_dataset_path = os.path.join(backend_dir, "simplified_dataset.json")
                demo_crew_dir = os.path.join(backend_dir, "demo_crew")
                demo_crew_output_path = os.path.join(demo_crew_dir, "processed_data.json")
                
                # Ensure demo_crew directory exists
                os.makedirs(os.path.dirname(demo_crew_output_path), exist_ok=True)
                
                # MODIFIED: Create a new dataset with only the current result instead of appending
                # This will clear any previous entries
                if "extracted_data" in result and "output" in result["extracted_data"] and "agent_type" in result["extracted_data"]:
                    new_dataset = {
                        "results": [
                            {
                                "output": result["extracted_data"]["output"],
                                "agent_type": result["extracted_data"]["agent_type"]
                            }
                        ]
                    }
                    
                    # Save the new dataset (overwriting any previous data)
                    with open(combined_dataset_path, "w") as f:
                        json.dump(new_dataset, f, indent=2)
                    
                    # Copy to the demo_crew directory
                    with open(demo_crew_output_path, "w") as f:
                        json.dump(new_dataset, f, indent=2)
                    
                    print(f"Simplified dataset created with new output and agent_type (previous data cleared)")
                else:
                    print("No output or agent_type found in the current result")
            else:
                print("OutputExtractorAgent not found in the MasterAgent's tools")
        
        except Exception as e:
            print(f"Error in OutputExtractorAgent processing: {str(e)}")
        
        return result_path
    
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
    # For demonstration, we'll check the processing result
    
    base_dir = ensure_upload_dirs()
    processed_dir = os.path.join(base_dir, "processed")
    
    # Check if result file exists (without extension)
    file_id_base = os.path.splitext(file_id)[0]
    result_files = [f for f in os.listdir(processed_dir) if f.startswith(file_id_base)]
    
    if result_files:
        # Find the simplified dataset file
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        simplified_dataset_path = os.path.join(backend_dir, "simplified_dataset.json")
        
        # If simplified dataset exists, use it instead of the full result
        if os.path.exists(simplified_dataset_path):
            try:
                with open(simplified_dataset_path, "r") as f:
                    simplified_data = json.load(f)
                
                # Return only the simplified data
                return {
                    "status": "completed",
                    "result": simplified_data,
                    "result_file": "/simplified_dataset.json"
                }
            except Exception as e:
                print(f"Error reading simplified dataset: {str(e)}")
        
        # Fallback to original result file if simplified dataset doesn't exist
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

@router.get("/dataset")
async def get_simplified_dataset():
    """
    Get the simplified dataset containing only output and agent_type fields
    """
    # Look for the simplified dataset in multiple locations
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "simplified_dataset.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "processed_data.json"),
    ]
    
    dataset = {"results": []}
    
    # Try each path until we find the dataset
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    dataset = json.load(f)
                break
            except Exception as e:
                print(f"Error reading {path}: {str(e)}")
    
    # Return only the output and agent_type fields for each result
    return dataset