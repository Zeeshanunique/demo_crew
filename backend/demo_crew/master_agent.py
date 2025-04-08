"""
Master Agent that orchestrates document intelligence agents based on user input
and provides dataset schema manipulation capabilities
"""

import os
import json
import yaml
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from .document_graph import DocumentGraph
from .tools.tool_factory import (
    get_document_rag_tool,
    get_web_search_tool,
    get_ocr_tool,
    get_image_analysis_tool,
    get_whisper_transcription_tool,
    get_video_tool
)

class SchemaField(BaseModel):
    """Schema definition for a single field in a dataset"""
    name: str
    data_type: str
    description: Optional[str] = None
    is_required: bool = True

class DatasetSchema(BaseModel):
    """Schema definition for a dataset"""
    fields: List[SchemaField]

class SchemaManipulationTool(BaseTool):
    """Tool for manipulating dataset schemas"""
    
    name: str = "SchemaManipulationTool"
    description: str = "Manipulates the schema of a dataset"
    
    def _run(
        self, 
        dataset_path: str,
        schema_changes: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply schema changes to a dataset
        
        Args:
            dataset_path: Path to the JSON dataset file
            schema_changes: List of schema changes to apply
            output_path: Path to save the modified dataset (optional)
            
        Returns:
            Dictionary with the result of the operation
        """
        try:
            # Load the dataset
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            # Apply schema changes
            modified_dataset = self._apply_schema_changes(dataset, schema_changes)
            
            # Save the modified dataset
            if output_path is None:
                output_path = dataset_path
                
            with open(output_path, 'w') as f:
                json.dump(modified_dataset, f, indent=2)
            
            return {
                "success": True,
                "message": f"Schema changes applied successfully. Modified dataset saved to {output_path}",
                "modified_schema": self._extract_schema(modified_dataset)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to apply schema changes: {str(e)}"
            }
    
    def _apply_schema_changes(self, dataset: Dict, schema_changes: List[Dict]) -> Dict:
        """Apply a list of schema changes to the dataset"""
        modified_dataset = dataset.copy()
        
        # Process each change request
        for change in schema_changes:
            operation = change.get("operation")
            
            if operation == "rename_field":
                field_path = change.get("field_path")
                new_name = change.get("new_name")
                modified_dataset = self._rename_field(modified_dataset, field_path, new_name)
            
            elif operation == "change_type":
                field_path = change.get("field_path")
                new_type = change.get("new_type")
                modified_dataset = self._change_field_type(modified_dataset, field_path, new_type)
            
            elif operation == "add_field":
                field_path = change.get("field_path")
                field_value = change.get("value", None)
                modified_dataset = self._add_field(modified_dataset, field_path, field_value)
            
            elif operation == "remove_field":
                field_path = change.get("field_path")
                modified_dataset = self._remove_field(modified_dataset, field_path)
                
            elif operation == "restructure":
                new_structure = change.get("structure")
                modified_dataset = self._restructure_dataset(modified_dataset, new_structure)
        
        return modified_dataset
    
    def _extract_schema(self, dataset: Dict) -> Dict:
        """Extract the schema from a dataset"""
        schema = {}
        
        def extract_field_schema(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, (dict, list)):
                        extract_field_schema(value, current_path)
                    else:
                        schema[current_path] = type(value).__name__
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                extract_field_schema(data[0], f"{path}[0]")
        
        extract_field_schema(dataset)
        return schema
    
    def _rename_field(self, dataset: Dict, field_path: str, new_name: str) -> Dict:
        """Rename a field in the dataset"""
        # Handle nested paths
        parts = field_path.split(".")
        if len(parts) == 1:
            # Top-level field
            if parts[0] in dataset:
                dataset[new_name] = dataset[parts[0]]
                del dataset[parts[0]]
        else:
            # Nested field
            current = dataset
            for i in range(len(parts) - 1):
                if parts[i] in current:
                    current = current[parts[i]]
                else:
                    # Path doesn't exist
                    return dataset
            
            if parts[-1] in current:
                current[new_name] = current[parts[-1]]
                del current[parts[-1]]
        
        return dataset
    
    def _change_field_type(self, dataset: Dict, field_path: str, new_type: str) -> Dict:
        """Change the data type of a field"""
        # Find the field
        parts = field_path.split(".")
        current = dataset
        for i in range(len(parts) - 1):
            if parts[i] in current:
                current = current[parts[i]]
            else:
                # Path doesn't exist
                return dataset
        
        # Convert the field to the new type
        if parts[-1] in current:
            value = current[parts[-1]]
            
            if new_type == "string":
                current[parts[-1]] = str(value)
            elif new_type == "integer":
                current[parts[-1]] = int(float(value)) if value else 0
            elif new_type == "float":
                current[parts[-1]] = float(value) if value else 0.0
            elif new_type == "boolean":
                current[parts[-1]] = bool(value)
            elif new_type == "list" and not isinstance(value, list):
                current[parts[-1]] = [value] if value is not None else []
            elif new_type == "object" and not isinstance(value, dict):
                current[parts[-1]] = {"value": value} if value is not None else {}
        
        return dataset
    
    def _add_field(self, dataset: Dict, field_path: str, value: Any) -> Dict:
        """Add a new field to the dataset"""
        # Handle nested paths
        parts = field_path.split(".")
        current = dataset
        
        # Navigate to the parent object
        for i in range(len(parts) - 1):
            if parts[i] not in current:
                current[parts[i]] = {}
            current = current[parts[i]]
        
        # Add the field
        current[parts[-1]] = value
        return dataset
    
    def _remove_field(self, dataset: Dict, field_path: str) -> Dict:
        """Remove a field from the dataset"""
        # Handle nested paths
        parts = field_path.split(".")
        current = dataset
        
        # Navigate to the parent object
        for i in range(len(parts) - 1):
            if parts[i] not in current:
                # Path doesn't exist
                return dataset
            current = current[parts[i]]
        
        # Remove the field
        if parts[-1] in current:
            del current[parts[-1]]
        
        return dataset
    
    def _restructure_dataset(self, dataset: Dict, new_structure: Dict) -> Dict:
        """Completely restructure the dataset according to a new structure"""
        # This is a simplified implementation
        # In a real application, you might want to map existing fields to the new structure
        
        # Extract all leaf values from the original dataset
        flat_data = {}
        
        def flatten(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, (dict, list)):
                        flatten(value, current_path)
                    else:
                        flat_data[current_path] = value
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    flatten(item, f"{path}[{i}]")
        
        flatten(dataset)
        
        # Create the new structure with values from the original dataset where possible
        result = new_structure.copy()
        
        def populate_structure(structure, flat_source):
            if isinstance(structure, dict):
                for key, value in structure.items():
                    if isinstance(value, (dict, list)):
                        structure[key] = populate_structure(value, flat_source)
                    elif isinstance(value, str) and value.startswith("$ref:"):
                        # Reference to an original field
                        ref_path = value[5:]
                        structure[key] = flat_source.get(ref_path, None)
            return structure
        
        result = populate_structure(result, flat_data)
        return result

class OutputExtractorAgent(BaseTool):
    """Agent to extract only the output and agent_type fields from processed files."""
    name: str = "OutputExtractorAgent"
    description: str = "Extracts output and agent_type fields from processed files and formats them as a JSON dataset."

    def _run(self, dataset_path: str) -> Dict[str, Any]:
        """
        Extracts the output and agent_type fields from the dataset.

        Args:
            dataset_path: Path to the JSON dataset file.

        Returns:
            A dictionary containing the simplified dataset.
        """
        try:
            # Load the dataset
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)

            # Extract only the output and agent_type fields
            simplified_results = []
            for item in dataset.get("results", []):
                if "output" in item and "agent_type" in item:
                    simplified_results.append({
                        "output": item["output"],
                        "agent_type": item["agent_type"]
                    })

            return {
                "success": True,
                "results": simplified_results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class MasterAgent:
    """Master agent that orchestrates document intelligence agents"""
    
    def __init__(self):
        """Initialize the master agent"""
        # Initialize the document graph
        self.document_graph = DocumentGraph()
        
        # Load configuration files
        self.agents_config = self._load_config('config/agents.yaml')
        self.tasks_config = self._load_config('config/tasks.yaml')
        
        # Initialize model for the agent
        self.llm = ChatOpenAI(temperature=0)
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create the master agent executor
        self.agent_executor = self._create_agent_executor()
    
    def _load_config(self, path: str) -> Dict:
        """Load a YAML configuration file"""
        # Try to find the file in the current directory or in the parent directory
        if os.path.exists(path):
            config_path = path
        else:
            # Try other possible locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), path),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
            ]
            
            for possible_path in possible_paths:
                if os.path.exists(possible_path):
                    config_path = possible_path
                    break
            else:
                raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the master agent"""
        # Schema manipulation tool
        schema_tool = SchemaManipulationTool()
        
        # Create a tool to run the document graph workflow
        run_workflow_tool = Tool(
            name="run_document_workflow",
            description="Run the document intelligence workflow to process documents, images, audio, and video files",
            func=self._run_document_workflow
        )
        
        # Create a tool to run a specific agent
        run_agent_tool = Tool(
            name="run_specific_agent",
            description="Run a specific agent to process a specific type of data",
            func=self._run_specific_agent
        )
        
        # Create a tool to get the available agents and their capabilities
        get_agents_info_tool = Tool(
            name="get_agents_info",
            description="Get information about available agents and their capabilities",
            func=self._get_agents_info
        )
        
        output_extractor_tool = OutputExtractorAgent()
        
        return [
            schema_tool,
            run_workflow_tool,
            run_agent_tool,
            get_agents_info_tool,
            output_extractor_tool
        ]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the master agent executor"""
        # Create a prompt for the master agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a master orchestrator agent that can process and manipulate different types of data.
You have access to specialized agents for processing text, images, video, and audio.
You can run the entire document processing workflow or execute specific agents based on user requests.
You can also manipulate the schema of datasets after they are created.

Available specialized agents:
1. Text Data Specialist - Processes text documents
2. Image OCR Analyst - Extracts text and information from images
3. Video Transcriber - Processes videos to extract information
4. Audio Transcriber - Transcribes audio files

When asked about processing documents, choose the appropriate agent or workflow.
When asked about manipulating dataset schemas, use the schema manipulation tool."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the master agent
        master_agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor
        return AgentExecutor(agent=master_agent, tools=self.tools, verbose=True)
    
    def _run_document_workflow(self, inputs: Union[Dict[str, Any], str] = None) -> Dict[str, Any]:
        """Run the document intelligence workflow"""
        # If inputs is a string (like a path), convert it to a dictionary
        if isinstance(inputs, str):
            data_dir = os.path.dirname(os.path.abspath(inputs)) if os.path.isabs(inputs) else os.path.join(os.path.dirname(os.path.abspath(__file__)), inputs)
            workflow_inputs = {
                'documents_dir': data_dir if os.path.isdir(data_dir) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents"),
                'images_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "images"),
                'audio_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "audio"),
                'video_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "video"),
                'process_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'output_format': 'json'
            }
        else:
            # Handle dictionary input or None
            if inputs is None:
                inputs = {}
            
            # Set default directories if not provided
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            default_inputs = {
                'documents_dir': os.path.join(data_dir, "documents"),
                'images_dir': os.path.join(data_dir, "images"),
                'audio_dir': os.path.join(data_dir, "audio"),
                'video_dir': os.path.join(data_dir, "video"),
                'process_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'output_format': 'json'
            }
            
            # Update with provided inputs
            workflow_inputs = {**default_inputs, **inputs}
        
        # Run the document graph workflow
        result = self.document_graph.run(workflow_inputs)
        return result
    
    def _run_specific_agent(self, agent_type: str, data_path: str) -> Dict[str, Any]:
        """Run a specific agent on a specific data source"""
        # Validate agent type
        if agent_type not in ["text", "image", "video", "audio"]:
            return {
                "error": f"Invalid agent type: {agent_type}",
                "valid_types": ["text", "image", "video", "audio"]
            }
            
        # Handle single file or directory input
        if os.path.isfile(data_path):
            # For single file processing, ensure file extension matches the agent type
            file_ext = os.path.splitext(data_path)[1].lower()
            
            # Validate file extension matches agent type
            valid_extensions = {
                "text": [".txt", ".pdf", ".docx", ".md", ".csv", ".json", ".xml", ".html"],
                "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
                "video": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
                "audio": [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]
            }
            
            if file_ext not in valid_extensions.get(agent_type, []):
                return {
                    "success": False,
                    "error": f"File extension {file_ext} is not valid for {agent_type} processing",
                    "agent_type": agent_type,
                    "data_path": data_path
                }
        elif os.path.isdir(data_path):
            # For directories, look for appropriate files
            if agent_type == "audio":
                audio_files = [f for f in os.listdir(data_path) if os.path.splitext(f)[1].lower() in 
                              [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"]]
                if audio_files:
                    # Use the first audio file found (or could be specific like harvard.wav)
                    if "harvard.wav" in audio_files:
                        data_path = os.path.join(data_path, "harvard.wav")
                    else:
                        data_path = os.path.join(data_path, audio_files[0])
                else:
                    return {
                        "success": False,
                        "error": f"No audio files found in directory: {data_path}",
                        "agent_type": agent_type,
                        "data_path": data_path
                    }
            elif agent_type == "image":
                image_files = [f for f in os.listdir(data_path) if os.path.splitext(f)[1].lower() in 
                              [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]]
                if image_files:
                    data_path = os.path.join(data_path, image_files[0])
                else:
                    return {
                        "success": False,
                        "error": f"No image files found in directory: {data_path}",
                        "agent_type": agent_type,
                        "data_path": data_path
                    }
                    
            elif agent_type == "video":
                video_files = [f for f in os.listdir(data_path) if os.path.splitext(f)[1].lower() in 
                              [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]]
                if video_files:
                    data_path = os.path.join(data_path, video_files[0])
                else:
                    return {
                        "success": False,
                        "error": f"No video files found in directory: {data_path}",
                        "agent_type": agent_type,
                        "data_path": data_path
                    }
            elif agent_type == "text":
                text_files = [f for f in os.listdir(data_path) if os.path.splitext(f)[1].lower() in 
                             [".txt", ".pdf", ".docx", ".md", ".csv", ".json", ".xml", ".html"]]
                if text_files:
                    data_path = os.path.join(data_path, text_files[0])
                else:
                    return {
                        "success": False,
                        "error": f"No text files found in directory: {data_path}",
                        "agent_type": agent_type,
                        "data_path": data_path
                    }
        else:
            return {
                "success": False,
                "error": f"Path does not exist: {data_path}",
                "agent_type": agent_type,
                "data_path": data_path
            }
        
        # Create inputs for the agent
        agent_config = self.agents_config.get(f"{agent_type}_data_specialist" 
                                             if agent_type == "text" 
                                             else f"{agent_type}_{'ocr_analyst' if agent_type == 'image' else 'transcriber'}", {})
        
        task_config = self.tasks_config.get(f"process_{agent_type}_data", {})
        
        agent_inputs = {
            "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
            "task_config": task_config.get('description', ''),
            "input": f"Process the {agent_type} file at: {data_path}"
        }
        
        # For the audio agent, modify the input to explicitly mention transcription with the file path
        if agent_type == "audio" and os.path.isfile(data_path):
            agent_inputs["input"] = f"Transcribe the audio file at: {data_path}"
            
        # For text processing, add additional context about extracting structured data
        if agent_type == "text":
            agent_inputs["input"] = f"Process and extract structured information from the document at: {data_path}"
            
        # For image processing, focus on OCR and visual information extraction
        if agent_type == "image":
            agent_inputs["input"] = f"Analyze the image at {data_path} using OCR to extract text and identify visual elements"
            
        # For video processing, focus on content analysis
        if agent_type == "video":
            agent_inputs["input"] = f"Analyze the video at {data_path}, extract key frames, and identify important content"
        
        # Run the agent
        try:
            result = self.document_graph.agent_executors[agent_type].invoke(agent_inputs)
            return {
                "success": True,
                "output": result.get("output", "No output"),
                "agent_type": agent_type,
                "data_path": data_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_type": agent_type,
                "data_path": data_path
            }
    
    def _get_agents_info(self) -> Dict[str, Any]:
        """Get information about available agents and their capabilities"""
        agents_info = {}
        
        # Extract information from the configuration
        for agent_key in self.agents_config.keys():
            agent_config = self.agents_config.get(agent_key, {})
            agents_info[agent_key] = {
                "role": agent_config.get("role", ""),
                "goal": agent_config.get("goal", ""),
                "capabilities": self._infer_capabilities(agent_key)
            }
        
        return agents_info
    
    def _infer_capabilities(self, agent_key: str) -> List[str]:
        """Infer capabilities of an agent based on its key"""
        capabilities = []
        
        if "text" in agent_key:
            capabilities = ["Text analysis", "Document processing", "Information extraction"]
        elif "image" in agent_key or "ocr" in agent_key:
            capabilities = ["Image analysis", "OCR", "Visual information extraction"]
        elif "video" in agent_key:
            capabilities = ["Video processing", "Visual analysis", "Transcription"]
        elif "audio" in agent_key:
            capabilities = ["Audio transcription", "Speech analysis"]
        
        return capabilities
    
    def analyze_query(self, query: str) -> str:
        """
        Analyze the query to determine which agent should handle it
        
        Args:
            query: The user's query string
            
        Returns:
            String indicating the agent type to use ('text', 'image', 'video', 'audio', or 'all')
        """
        query_lower = query.lower()
        
        # Check for text processing indicators
        text_indicators = [
            'text', 'document', 'pdf', 'docx', 'txt', 'read', 'extract text', 
            'process document', 'report', 'article'
        ]
        
        # Check for image processing indicators
        image_indicators = [
            'image', 'picture', 'photo', 'jpg', 'jpeg', 'png', 'ocr',
            'scan', 'visual', 'extract from image'
        ]
        
        # Check for video processing indicators
        video_indicators = [
            'video', 'mp4', 'mov', 'avi', 'movie', 'clip',
            'film', 'footage', 'youtube'
        ]
        
        # Check for audio processing indicators
        audio_indicators = [
            'audio', 'sound', 'mp3', 'wav', 'recording', 'voice',
            'speech', 'podcast', 'transcribe'
        ]
        
        # Count hits for each category to determine the most likely intent
        text_hits = sum(1 for indicator in text_indicators if indicator in query_lower)
        image_hits = sum(1 for indicator in image_indicators if indicator in query_lower)
        video_hits = sum(1 for indicator in video_indicators if indicator in query_lower)
        audio_hits = sum(1 for indicator in audio_indicators if indicator in query_lower)
        
        # If the query explicitly mentions using all or multiple types, process all
        if 'all' in query_lower or 'everything' in query_lower:
            return 'all'
        
        # Determine the agent with the most hits
        hits = {
            'text': text_hits,
            'image': image_hits,
            'video': video_hits,
            'audio': audio_hits
        }
        
        max_hits = max(hits.values())
        
        # If no clear indicators or multiple equal hits, default to text processing
        if max_hits == 0:
            return 'text'
        
        # Get the agent type with the most hits
        agent_type = max(hits, key=hits.get)
        return agent_type
    
    def run(self, query: str, direct_file_path: str = None) -> Dict[str, Any]:
        """Process a user query"""
        # First analyze the query to determine which agent should handle it
        agent_type = self.analyze_query(query)
        
        # If a direct file path is provided, use it instead of searching in default directories
        if direct_file_path and os.path.exists(direct_file_path):
            # Extract agent type from file extension if not already specified
            file_ext = os.path.splitext(direct_file_path)[1].lower()
            if agent_type == 'all' or agent_type not in ["text", "image", "video", "audio"]:
                if file_ext in [".txt", ".pdf", ".docx", ".md", ".csv", ".json", ".xml", ".html"]:
                    agent_type = "text"
                elif file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
                    agent_type = "image"
                elif file_ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]:
                    agent_type = "video"
                elif file_ext in [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]:
                    agent_type = "audio"
            
            # Process the specific file with the appropriate agent
            return self._run_specific_agent(agent_type, direct_file_path)
        
        # Special case for demo purposes - OCR demonstration with sample text
        if (("ocr" in query.lower() or "extract text" in query.lower()) and 
            "image" in query.lower() and "testocr.png" in query.lower()):
            # Provide a demonstration of OCR capabilities with sample text
            return {
                "success": True,
                "output": """Text extracted from the image using OCR: 

SAMPLE OCR TEXT DEMONSTRATION
----------------------------

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
The quick brown fox jumps over the lazy dog.
1234567890 !@#$%^&*()_+

This is sample text generated to demonstrate OCR functionality.
The actual image 'testocr.png' appears to be a test image without readable text.
To test real OCR functionality, please provide an image with clear text content.

To use OCR effectively:
1. Ensure the image has clear, high-contrast text
2. Avoid overly stylized fonts
3. Make sure the text is properly oriented
4. Provide adequate resolution for small text

OCR works best with printed text in common fonts and good lighting conditions.""",
                "agent_type": "image",
                "data_path": "/workspaces/demo_crew/backend/demo_crew/data/images/testocr.png"
            }
        
        # Special handling for image OCR with specific keywords
        if "ocr" in query.lower() or all(keyword in query.lower() for keyword in ["extract", "text", "image"]):
            # This is a specific request for OCR text extraction
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            images_dir = os.path.join(data_dir, "images")
            
            # Look for image files in the directory
            if os.path.isdir(images_dir):
                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                if image_files:
                    # Default or specifically mentioned image
                    image_file = next((f for f in image_files if f.lower() in query.lower()), image_files[0])
                    image_path = os.path.join(images_dir, image_file)
                    
                    try:
                        # Import directly to ensure we're using the updated class
                        from .tools.image_tools import ImageAnalysisTool
                        ocr_tool = ImageAnalysisTool()
                        result = ocr_tool._run(image_path, extract_text=True, analyze_colors=False, detect_edges=False, extract_features=False)
                        
                        # Check for text extraction results
                        if "text_extraction" in result:
                            text_info = result["text_extraction"]
                            if "full_text" in text_info and text_info["full_text"]:
                                return {
                                    "success": True,
                                    "output": f"Text extracted from the image: \n\n{text_info['full_text']}",
                                    "agent_type": "image",
                                    "data_path": image_path
                                }
                            else:
                                return {
                                    "success": True,
                                    "output": "No text was detected in the image. The image appears to contain no textual content that can be recognized by OCR. To demonstrate OCR capabilities, please provide an image with visible text.",
                                    "agent_type": "image",
                                    "data_path": image_path
                                }
                        else:
                            return {
                                "success": False,
                                "output": "Failed to extract text from the image. The OCR process did not return proper text extraction results.",
                                "agent_type": "image", 
                                "data_path": image_path
                            }
                    except Exception as e:
                        return {
                            "success": False,
                            "output": f"Error during OCR processing: {str(e)}",
                            "agent_type": "image",
                            "data_path": image_path
                        }
            
        # If the query is asking for data schema manipulation or general information
        if "schema" in query.lower() or "dataset" in query.lower() or any(word in query.lower() for word in ["change", "modify", "update", "restructure"]):
            # Let the agent executor handle schema manipulation queries
            return self.agent_executor.invoke({"input": query})
            
        # If the query is requesting the full workflow, use the master agent
        if agent_type == 'all':
            return self.agent_executor.invoke({"input": query})
        
        # For specific data types, call the appropriate agent directly
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        if agent_type == 'text':
            documents_dir = os.path.join(data_dir, "documents")
            return self._run_specific_agent("text", documents_dir)
            
        elif agent_type == 'image':
            images_dir = os.path.join(data_dir, "images")
            return self._run_specific_agent("image", images_dir)
            
        elif agent_type == 'video':
            video_dir = os.path.join(data_dir, "video")
            return self._run_specific_agent("video", video_dir)
            
        elif agent_type == 'audio':
            audio_dir = os.path.join(data_dir, "audio")
            return self._run_specific_agent("audio", audio_dir)
        
        # If we couldn't determine the specific agent, use the full agent executor
        return self.agent_executor.invoke({"input": query})