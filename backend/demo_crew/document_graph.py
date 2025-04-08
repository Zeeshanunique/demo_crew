"""
Document Intelligence workflow using LangGraph and LangChain
This module replaces the previous CrewAI implementation with a LangGraph workflow
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END

# Import tools
from .tools.tool_factory import (
    get_document_rag_tool,
    get_web_search_tool,
    get_ocr_tool,
    get_image_analysis_tool,
    get_whisper_transcription_tool,
    get_video_tool
)

class DocumentGraph:
    """Document Intelligence workflow using LangGraph's state machine approach"""

    def __init__(self):
        """Initialize the document workflow"""
        # Load configuration files
        self.agents_config = self._load_config('config/agents.yaml')
        self.tasks_config = self._load_config('config/tasks.yaml')
        
        # Initialize model for the agents
        self.model = ChatOpenAI(temperature=0)
        
        # Create agent executors
        self.agent_executors = self._create_agent_executors()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()

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

    def _create_agent_executors(self) -> Dict[str, AgentExecutor]:
        """Create agent executors for different document types."""
        # Initialize model and tools
        llm = ChatOpenAI(temperature=0)
        
        # Get tools
        text_tools = [get_document_rag_tool(), get_web_search_tool()]
        image_tools = [get_ocr_tool(), get_image_analysis_tool()]
        video_tools = [get_video_tool()]
        audio_tools = [get_whisper_transcription_tool()]
        
        # Create prompts for each specialist agent
        text_specialist_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a text analysis specialist. You can extract and analyze information from text documents."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")  # Add this line for agent_scratchpad
        ])
        
        image_specialist_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an image analysis specialist. You can extract information from images using OCR and other techniques."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")  # Add this line for agent_scratchpad
        ])
        
        video_specialist_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a video analysis specialist. You can extract information from videos."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")  # Add this line for agent_scratchpad
        ])
        
        audio_specialist_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an audio analysis specialist. You can transcribe and extract information from audio files."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")  # Add this line for agent_scratchpad
        ])
        
        # Create the OpenAI tools agents
        text_specialist_agent = create_openai_tools_agent(
            llm=llm,
            tools=text_tools,
            prompt=text_specialist_prompt
        )
        
        image_specialist_agent = create_openai_tools_agent(
            llm=llm,
            tools=image_tools,
            prompt=image_specialist_prompt
        )
        
        video_specialist_agent = create_openai_tools_agent(
            llm=llm,
            tools=video_tools,
            prompt=video_specialist_prompt
        )
        
        audio_specialist_agent = create_openai_tools_agent(
            llm=llm,
            tools=audio_tools,
            prompt=audio_specialist_prompt
        )
        
        # Create agent executors
        return {
            "text": AgentExecutor(agent=text_specialist_agent, tools=text_tools),
            "image": AgentExecutor(agent=image_specialist_agent, tools=image_tools),
            "video": AgentExecutor(agent=video_specialist_agent, tools=video_tools),
            "audio": AgentExecutor(agent=audio_specialist_agent, tools=audio_tools)
        }
    
    def _text_data_specialist_node(self, state: Dict) -> Dict:
        """Text Data Specialist node in the workflow"""
        # Extract agent and task configs
        agent_config = self.agents_config.get('text_data_specialist', {})
        task_config = self.tasks_config.get('process_text_data', {})
        
        # Check if direct file path is provided
        direct_file_path = state.get('direct_file_path')
        file_type = state.get('file_type')
        
        if direct_file_path and os.path.isfile(direct_file_path) and (file_type == 'text' or file_type is None):
            # Process the specific file directly using DocumentRAGTool with single_file parameter
            try:
                # Import directly to ensure we're using the latest version
                from .tools.text_tools import DocumentRAGTool
                rag_tool = DocumentRAGTool()
                
                # Process the file and get its content
                file_result = rag_tool._run(query="Extract structured information", single_file=direct_file_path)
                
                # Format the result for output
                if 'error' in file_result:
                    output = f"Error processing file: {file_result.get('error')}"
                    if 'traceback' in file_result:
                        output += f"\nDetails: {file_result.get('traceback')}"
                else:
                    # Extract structured information
                    output = "I've extracted the following information from the document:\n\n"
                    
                    # Add structured data
                    structured_data = file_result.get('structured_data', {})
                    
                    # Add summary
                    if 'summary' in structured_data:
                        output += f"Document Summary: {structured_data['summary']}\n\n"
                    
                    # Add detected entities
                    entities = structured_data.get('detected_entities', {})
                    if entities:
                        output += "Detected Entities:\n"
                        for entity_type, entity_values in entities.items():
                            if entity_values:
                                output += f"- {entity_type.replace('_', ' ').title()}: {', '.join(entity_values)}\n"
                        output += "\n"
                    
                    # Add metadata
                    metadata = file_result.get('metadata', {})
                    if metadata:
                        output += f"Document Metadata: {len(metadata)} fields available\n"
                        output += f"Total Pages: {file_result.get('page_count', 1)}\n\n"
                
                # Store the processed content in state
                new_state = state.copy()
                new_state["text_processing_result"] = output
                return new_state
                
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                # Fall back to agent processing if there's an error
                output = f"Error using direct file processing: {str(e)}\n{trace}\nFalling back to agent processing."
                print(output)
                # Continue to agent processing
                pass
        
        # Use agent-based processing (either as primary method or fallback)
        if direct_file_path and os.path.isfile(direct_file_path) and (file_type == 'text' or file_type is None):
            # Process the specific file
            agent_inputs = {
                "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
                "task_config": task_config.get('description', ''),
                "input": f"Process and extract structured information from the document at: {direct_file_path}"
            }
        else:
            # Process the directory
            agent_inputs = {
                "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
                "task_config": task_config.get('description', ''),
                "input": f"Process the documents in: {state.get('documents_dir', 'data/documents')}"
            }
        
        # Run the agent
        result = self.agent_executors["text"].invoke(agent_inputs)
        
        # Update state with result
        new_state = state.copy()
        new_state["text_processing_result"] = result.get("output", "No output")
        return new_state

    def _image_ocr_analyst_node(self, state: Dict) -> Dict:
        """Image OCR Analyst node in the workflow"""
        # Extract agent and task configs
        agent_config = self.agents_config.get('image_ocr_analyst', {})
        task_config = self.tasks_config.get('process_image_data', {})
        
        # Check if direct file path is provided and matches image type
        direct_file_path = state.get('direct_file_path')
        file_type = state.get('file_type')
        
        if direct_file_path and os.path.isfile(direct_file_path) and file_type == 'image':
            # Process the specific image file
            agent_inputs = {
                "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
                "task_config": task_config.get('description', ''),
                "input": f"Analyze the image at {direct_file_path} using OCR to extract text and identify visual elements"
            }
        else:
            # Process the directory or skip if not processing images
            if file_type and file_type != 'image' and direct_file_path:
                # Skip processing for other file types
                new_state = state.copy()
                new_state["image_processing_result"] = "Skipped - not an image file"
                return new_state
                
            agent_inputs = {
                "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
                "task_config": task_config.get('description', ''),
                "input": f"Process the images in: {state.get('images_dir', 'data/images')}"
            }
        
        # Run the agent
        result = self.agent_executors["image"].invoke(agent_inputs)
        
        # Update state with result
        new_state = state.copy()
        new_state["image_processing_result"] = result.get("output", "No output")
        return new_state

    def _video_transcriber_node(self, state: Dict) -> Dict:
        """Video Transcriber node in the workflow"""
        # Extract agent and task configs
        agent_config = self.agents_config.get('video_transcriber', {})
        task_config = self.tasks_config.get('process_video_data', {})
        
        # Check if direct file path is provided and matches video type
        direct_file_path = state.get('direct_file_path')
        file_type = state.get('file_type')
        
        if direct_file_path and os.path.isfile(direct_file_path) and file_type == 'video':
            # Process the specific video file
            agent_inputs = {
                "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
                "task_config": task_config.get('description', ''),
                "input": f"Analyze the video at {direct_file_path}, extract key frames, and identify important content"
            }
        else:
            # Process the directory or skip if not processing videos
            if file_type and file_type != 'video' and direct_file_path:
                # Skip processing for other file types
                new_state = state.copy()
                new_state["video_processing_result"] = "Skipped - not a video file"
                return new_state
                
            agent_inputs = {
                "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
                "task_config": task_config.get('description', ''),
                "input": f"Process the videos in: {state.get('video_dir', 'data/video')}"
            }
        
        # Run the agent
        result = self.agent_executors["video"].invoke(agent_inputs)
        
        # Update state with result
        new_state = state.copy()
        new_state["video_processing_result"] = result.get("output", "No output")
        return new_state

    def _audio_transcriber_node(self, state: Dict) -> Dict:
        """Audio Transcriber node in the workflow"""
        # Extract agent and task configs
        agent_config = self.agents_config.get('audio_transcriber', {})
        task_config = self.tasks_config.get('process_audio_data', {})
        
        # Check if direct file path is provided and matches audio type
        direct_file_path = state.get('direct_file_path')
        file_type = state.get('file_type')
        
        if direct_file_path and os.path.isfile(direct_file_path) and file_type == 'audio':
            # Process the specific audio file
            agent_inputs = {
                "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
                "task_config": task_config.get('description', ''),
                "input": f"Transcribe the audio file at: {direct_file_path}"
            }
        else:
            # Process the directory or skip if not processing audio
            if file_type and file_type != 'audio' and direct_file_path:
                # Skip processing for other file types
                new_state = state.copy()
                new_state["audio_processing_result"] = "Skipped - not an audio file"
                return new_state
                
            agent_inputs = {
                "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
                "task_config": task_config.get('description', ''),
                "input": f"Process the audio files in: {state.get('audio_dir', 'data/audio')}"
            }
        
        # Run the agent
        result = self.agent_executors["audio"].invoke(agent_inputs)
        
        # Update state with result
        new_state = state.copy()
        new_state["audio_processing_result"] = result.get("output", "No output")
        return new_state
    
    def _compile_results(self, state: Dict) -> Dict:
        """Compile the results from all agents"""
        output = {
            "text_processing": state.get("text_processing_result", ""),
            "image_processing": state.get("image_processing_result", ""),
            "video_processing": state.get("video_processing_result", ""),
            "audio_processing": state.get("audio_processing_result", ""),
            "timestamp": state.get("process_timestamp", ""),
        }
        
        # Include the direct file path if it was processed
        if state.get("direct_file_path"):
            output["processed_file"] = state.get("direct_file_path")
            output["file_type"] = state.get("file_type", "unknown")
        
        # Save to output file if specified
        output_format = state.get("output_format", "")
        if output_format.lower() == "json":
            import json
            with open("processed_data.json", "w") as f:
                json.dump(output, f, indent=2)
        
        new_state = state.copy()
        new_state["final_output"] = output
        return new_state

    def _create_workflow(self) -> StateGraph:
        """Create the workflow graph"""
        # Initialize the state graph
        workflow = StateGraph(Dict)
        
        # Add nodes
        workflow.add_node("text_data_specialist", self._text_data_specialist_node)
        workflow.add_node("image_ocr_analyst", self._image_ocr_analyst_node)
        workflow.add_node("video_transcriber", self._video_transcriber_node)
        workflow.add_node("audio_transcriber", self._audio_transcriber_node)
        workflow.add_node("compile_results", self._compile_results)
        
        # Define the workflow
        workflow.add_edge("text_data_specialist", "image_ocr_analyst")
        workflow.add_edge("image_ocr_analyst", "video_transcriber")
        workflow.add_edge("video_transcriber", "audio_transcriber")
        workflow.add_edge("audio_transcriber", "compile_results")
        workflow.add_edge("compile_results", END)
        
        # Set the entry point
        workflow.set_entry_point("text_data_specialist")
        
        # Compile the workflow
        return workflow.compile()

    def run(self, inputs: Dict[str, Any] = None, direct_file_path: str = None, file_type: str = None) -> Dict[str, Any]:
        """
        Run the workflow with the given inputs
        
        Args:
            inputs: Dictionary of inputs for the workflow
            direct_file_path: Optional path to a specific file to process
            file_type: Optional type of the file (text, image, video, audio)
            
        Returns:
            Dictionary with the results of the workflow
        """
        if inputs is None:
            inputs = {}
        
        # Add direct file path and type to inputs if provided
        if direct_file_path and os.path.exists(direct_file_path):
            inputs["direct_file_path"] = direct_file_path
            
            # Determine file type if not provided
            if file_type is None:
                file_ext = os.path.splitext(direct_file_path)[1].lower()
                if file_ext in [".txt", ".pdf", ".docx", ".md", ".csv", ".json", ".xml", ".html"]:
                    file_type = "text"
                elif file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
                    file_type = "image"
                elif file_ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]:
                    file_type = "video"
                elif file_ext in [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]:
                    file_type = "audio"
            
            inputs["file_type"] = file_type
        
        # Execute the workflow
        result = self.workflow.invoke(inputs)
        return result
