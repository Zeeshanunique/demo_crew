"""
Document Intelligence workflow using LangGraph and LangChain
This module replaces the previous CrewAI implementation with a LangGraph workflow
"""

import os
import yaml
import time
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from langsmith import Client
from langsmith.run_trees import RunTree

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
        
        # Set up LangSmith tracing if environment variables are available
        self.tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        if self.tracing_enabled:
            self.langsmith_client = Client(
                api_key=os.getenv("LANGCHAIN_API_KEY"),
                api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            )
            print("LangSmith tracing enabled. Runs will be logged to LangSmith.")
        else:
            print("LangSmith tracing disabled. Set LANGCHAIN_TRACING_V2=true to enable.")
            
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
            # Try direct OCR processing first
            try:
                from .tools.image_tools import MistralOCRTool, ImageAnalysisTool
                
                # Check if we should use ground truth for accuracy evaluation
                ground_truth = None
                if "image_ground_truth" in state:
                    ground_truth = state.get("image_ground_truth")
                else:
                    # Use predefined ground truth for common test images
                    image_name = os.path.basename(direct_file_path).lower()
                    if "ocr_test" in image_name or "text_sample" in image_name:
                        ground_truth = "This is a Iot of 12 point text to test the ocr code and see if it works on all types of file format.. The quick brown dog jumped over the lazy fox. The quick brown dog jumped. over the lazy fox. The quick brown dog jumped over the lazy fox. The quick brown dog jumped over the lazy fox."
                
                # First run OCR if ground truth is available to get accuracy
                if ground_truth:
                    ocr_tool = MistralOCRTool()
                    ocr_result = ocr_tool._run(image_path=direct_file_path, ground_truth=ground_truth)
                    
                    # Store accuracy metrics in state if available
                    if 'accuracy' in ocr_result:
                        new_state = state.copy()
                        new_state["image_accuracy"] = ocr_result['accuracy']
                
                # Then run full image analysis
                image_tool = ImageAnalysisTool()
                image_result = image_tool._run(
                    direct_file_path, 
                    extract_text=True, 
                    analyze_colors=True,
                    detect_edges=False,
                    extract_features=True
                )
                
                if 'error' not in image_result:
                    # Format the OCR result for output
                    output = "I've analyzed the image and extracted the following information:\n\n"
                    
                    # Add text extraction results
                    if 'text_extraction' in image_result and 'full_text' in image_result['text_extraction']:
                        extracted_text = image_result['text_extraction']['full_text']
                        if extracted_text:
                            output += f"## Extracted Text\n{extracted_text}\n\n"
                        else:
                            output += "No text was detected in the image.\n\n"
                    
                    # Add color analysis
                    if 'color_analysis' in image_result and 'dominant_colors' in image_result['color_analysis']:
                        colors = image_result['color_analysis']['dominant_colors']
                        output += "## Color Analysis\n"
                        output += "Dominant colors in the image:\n"
                        for idx, color in enumerate(colors[:3], 1):
                            percentage = round(color['frequency'] * 100, 1)
                            output += f"{idx}. {color['hex']} ({percentage}% of image)\n"
                        output += "\n"
                    
                    # Add image features
                    if 'image_features' in image_result:
                        features = image_result['image_features']
                        output += "## Image Properties\n"
                        if 'dimensions' in features:
                            dims = features['dimensions']
                            output += f"Dimensions: {dims.get('width', 0)}x{dims.get('height', 0)}\n"
                            output += f"Aspect ratio: {dims.get('aspect_ratio', 0):.2f}\n"
                        if 'brightness' in features:
                            output += f"Brightness: {features['brightness']:.1f}/255\n"
                        if 'contrast' in features:
                            output += f"Contrast: {features['contrast']:.1f}\n"
                        output += f"File size: {features.get('file_size_kb', 0):.1f} KB\n\n"
                    
                    # Update state with result
                    new_state = state.copy()
                    new_state["image_processing_result"] = output
                    return new_state
                
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                print(f"Error in direct image processing: {str(e)}\n{trace}")
                # Continue to agent-based processing
            
            # Process the specific image file using agent
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
            # Process the specific video file directly first
            try:
                print(f"Processing video file: {direct_file_path}")
                from .tools.video_tools import VideoPyDLPTool
                video_tool = VideoPyDLPTool()
                # Initialize output variable before using it
                output = "I've analyzed the video and extracted the following information:\n\n"
                # Explicitly set extract_audio=True to ensure audio extraction
                video_result = video_tool._run(direct_file_path, extract_frames=True, extract_audio=True)
                
                if 'error' not in video_result:
                    
                    # Add metadata about the video
                    if 'metadata' in video_result:
                        meta = video_result['metadata']
                        output += f"## Video Properties\n"
                        output += f"Duration: {meta.get('duration_seconds', 0):.2f} seconds\n"
                        output += f"Resolution: {meta.get('width', 0)}x{meta.get('height', 0)}\n"
                        output += f"Frame rate: {meta.get('fps', 0):.2f} fps\n"
                        output += f"Total frames: {meta.get('frame_count', 0)}\n\n"
                    
                    # Add audio transcription if available
                    if 'audio' in video_result and 'transcript' in video_result['audio'] and video_result['audio']['transcript']:
                        output += f"## Audio Transcript\n"
                        output += video_result['audio']['transcript']
                        output += "\n\n"
                        
                        # Add language detection if available
                        if 'language' in video_result['audio']:
                            output += f"Detected language: {video_result['audio']['language']}\n\n"
                    
                    # Add key frames info if available
                    if 'frames' in video_result and video_result['frames']:
                        output += f"## Key Frames\n"
                        output += f"Extracted {len(video_result['frames'])} key frames at the following timestamps:\n"
                        for i, frame in enumerate(video_result['frames']):
                            output += f"{i+1}. {frame.get('timestamp', 0):.2f}s\n"
                        output += "\n"
                    
                    # Update state with result
                    new_state = state.copy()
                    new_state["video_processing_result"] = output
                    return new_state
                    
                else:
                    print(f"Error in direct video processing: {video_result.get('error', 'Unknown error')}")
                    if 'traceback' in video_result:
                        print(video_result['traceback'])
                    # Continue to agent-based processing
                
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                print(f"Error in direct video processing: {str(e)}\n{trace}")
                # Continue to agent-based processing
            
            # Process the specific video file using agent as fallback
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
            # Try direct audio processing first
            try:
                from .tools.audio_tools import WhisperTranscriptionTool
                
                # Check if we should use ground truth for accuracy evaluation
                ground_truth = None
                if "audio_ground_truth" in state:
                    ground_truth = state.get("audio_ground_truth")
                else:
                    # Use predefined ground truth for common test audio files
                    audio_name = os.path.basename(direct_file_path).lower()
                    if "test_sample" in audio_name or "harvard" in audio_name:
                        ground_truth = """1. "The stale smell of old beer lingers."
2. "It takes heat to bring out the odor."
3. "A cold dip restores health and zest."
4. "A salt pickle tastes fine with ham."
5. "Tacos al pastor are my favorite."
6. "A zestful food is the hot cross bun."

Language: English"""
                
                # Run transcription with ground truth for accuracy calculation if available
                audio_tool = WhisperTranscriptionTool(model_size="base")
                result = audio_tool._run(direct_file_path, ground_truth=ground_truth)
                
                # Store accuracy metrics in state if available
                if 'accuracy' in result:
                    new_state = state.copy()
                    new_state["audio_accuracy"] = result['accuracy']
                
                if 'error' not in result:
                    # Format the result for output
                    output = "I've transcribed the audio and extracted the following information:\n\n"
                    
                    # Add the transcription
                    if 'transcription' in result:
                        output += f"## Audio Transcript\n{result['transcription']}\n\n"
                    
                    # Add language detection if available
                    if 'language' in result:
                        output += f"Detected language: {result['language']}\n\n"
                    
                    # Add segments with timestamps if available
                    if 'segments' in result and result['segments']:
                        output += "## Segment Details\n"
                        for i, segment in enumerate(result['segments'][:5]):  # Show first 5 segments only
                            output += f"{i+1}. ({segment['start']:.2f}s - {segment['end']:.2f}s): {segment['text']}\n"
                        if len(result['segments']) > 5:
                            output += f"... {len(result['segments']) - 5} more segments\n"
                        output += "\n"
                    
                    # Update state with result
                    new_state = state.copy()
                    new_state["audio_processing_result"] = output
                    return new_state
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                print(f"Error in direct audio processing: {str(e)}\n{trace}")
                # Continue to agent-based processing
            
            # Process the specific audio file using agent
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
        """Compile the results from all agents and display accuracy metrics if available"""
        output = {
            "text_processing": state.get("text_processing_result", ""),
            "image_processing": state.get("image_processing_result", ""),
            "video_processing": state.get("video_processing_result", ""),
            "audio_processing": state.get("audio_processing_result", ""),
            "timestamp": state.get("process_timestamp", ""),
        }
        
        # Calculate accuracy metrics based on result quality
        accuracy_metrics = {
            "text_accuracy": 0.0,
            "image_accuracy": 0.0,
            "video_accuracy": 0.0,
            "audio_accuracy": 0.0,
            "overall_accuracy": 0.0
        }
        
        # Include the direct file path if it was processed
        if state.get("direct_file_path"):
            output["processed_file"] = state.get("direct_file_path")
            output["file_type"] = state.get("file_type", "unknown")
            
            # Display accuracy metrics in terminal if they exist
            file_type = state.get("file_type", "")
            if file_type == "image" and hasattr(state, "image_accuracy"):
                accuracy = state.get("image_accuracy", {})
                if accuracy:
                    print("\n===== FINAL IMAGE PROCESSING ACCURACY =====")
                    print(f"Overall Accuracy: {accuracy.get('overall_accuracy', 0):.2%}")
                    print(f"Confidence Level: {accuracy.get('confidence_level', 'Unknown')}")
                    print("===========================================\n")
                    accuracy_metrics["image_accuracy"] = accuracy.get("overall_accuracy", 0.0)
            
            elif file_type == "audio" and hasattr(state, "audio_accuracy"):
                accuracy = state.get("audio_accuracy", {})
                if accuracy:
                    print("\n===== FINAL AUDIO PROCESSING ACCURACY =====")
                    print(f"Overall Accuracy: {accuracy.get('overall_accuracy', 0):.2%}")
                    print(f"Word Match Ratio: {accuracy.get('word_match_ratio', 0):.2%}")
                    print(f"Confidence Level: {accuracy.get('confidence_level', 'Unknown')}")
                    print("===========================================\n")
                    accuracy_metrics["audio_accuracy"] = accuracy.get("overall_accuracy", 0.0)
        
        # Add accuracy metrics to output
        output["accuracy_metrics"] = accuracy_metrics
        
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
