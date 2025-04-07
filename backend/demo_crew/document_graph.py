"""
Document Intelligence workflow using LangGraph and LangChain
This module replaces the previous CrewAI implementation with a LangGraph workflow
"""

import os
import yaml
from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate
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
        """Create LangChain agent executors for each role"""
        agent_executors = {}
        
        # Text Data Specialist
        text_tools = [get_document_rag_tool(), get_web_search_tool()]
        text_specialist_prompt = ChatPromptTemplate.from_template(
            """You are a Text Data Specialist. {agent_config}
            
            Your task is to: {task_config}
            
            Use the provided tools to accomplish your task.
            
            {input}
            """
        )
        text_specialist_agent = create_openai_tools_agent(
            self.model,
            text_tools,
            text_specialist_prompt
        )
        agent_executors["text_data_specialist"] = AgentExecutor(
            agent=text_specialist_agent,
            tools=text_tools,
            verbose=True
        )
        
        # Image OCR Analyst
        image_tools = [get_ocr_tool(), get_image_analysis_tool()]
        image_analyst_prompt = ChatPromptTemplate.from_template(
            """You are an OCR Image Analyst. {agent_config}
            
            Your task is to: {task_config}
            
            Use the provided tools to accomplish your task.
            
            {input}
            """
        )
        image_analyst_agent = create_openai_tools_agent(
            self.model,
            image_tools,
            image_analyst_prompt
        )
        agent_executors["image_ocr_analyst"] = AgentExecutor(
            agent=image_analyst_agent,
            tools=image_tools,
            verbose=True
        )
        
        # Video Transcriber
        video_tools = [get_video_tool()]
        video_specialist_prompt = ChatPromptTemplate.from_template(
            """You are a Video-to-Text Transcriber. {agent_config}
            
            Your task is to: {task_config}
            
            Use the provided tools to accomplish your task.
            
            {input}
            """
        )
        video_specialist_agent = create_openai_tools_agent(
            self.model,
            video_tools,
            video_specialist_prompt
        )
        agent_executors["video_transcriber"] = AgentExecutor(
            agent=video_specialist_agent,
            tools=video_tools,
            verbose=True
        )
        
        # Audio Transcriber
        audio_tools = [get_whisper_transcription_tool()]
        audio_specialist_prompt = ChatPromptTemplate.from_template(
            """You are an Audio Transcriber. {agent_config}
            
            Your task is to: {task_config}
            
            Use the provided tools to accomplish your task.
            
            {input}
            """
        )
        audio_specialist_agent = create_openai_tools_agent(
            self.model,
            audio_tools,
            audio_specialist_prompt
        )
        agent_executors["audio_transcriber"] = AgentExecutor(
            agent=audio_specialist_agent,
            tools=audio_tools,
            verbose=True
        )
        
        return agent_executors
    
    def _text_data_specialist_node(self, state: Dict) -> Dict:
        """Text Data Specialist node in the workflow"""
        # Extract agent and task configs
        agent_config = self.agents_config.get('text_data_specialist', {})
        task_config = self.tasks_config.get('process_text_data', {})
        
        # Prepare inputs
        agent_inputs = {
            "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
            "task_config": task_config.get('description', ''),
            "input": f"Process the documents in: {state.get('documents_dir', 'data/documents')}"
        }
        
        # Run the agent
        result = self.agent_executors["text_data_specialist"].invoke(agent_inputs)
        
        # Update state with result
        new_state = state.copy()
        new_state["text_processing_result"] = result.get("output", "No output")
        return new_state

    def _image_ocr_analyst_node(self, state: Dict) -> Dict:
        """Image OCR Analyst node in the workflow"""
        # Extract agent and task configs
        agent_config = self.agents_config.get('image_ocr_analyst', {})
        task_config = self.tasks_config.get('process_image_data', {})
        
        # Prepare inputs
        agent_inputs = {
            "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
            "task_config": task_config.get('description', ''),
            "input": f"Process the images in: {state.get('images_dir', 'data/images')}"
        }
        
        # Run the agent
        result = self.agent_executors["image_ocr_analyst"].invoke(agent_inputs)
        
        # Update state with result
        new_state = state.copy()
        new_state["image_processing_result"] = result.get("output", "No output")
        return new_state

    def _video_transcriber_node(self, state: Dict) -> Dict:
        """Video Transcriber node in the workflow"""
        # Extract agent and task configs
        agent_config = self.agents_config.get('video_transcriber', {})
        task_config = self.tasks_config.get('process_video_data', {})
        
        # Prepare inputs
        agent_inputs = {
            "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
            "task_config": task_config.get('description', ''),
            "input": f"Process the videos in: {state.get('video_dir', 'data/video')}"
        }
        
        # Run the agent
        result = self.agent_executors["video_transcriber"].invoke(agent_inputs)
        
        # Update state with result
        new_state = state.copy()
        new_state["video_processing_result"] = result.get("output", "No output")
        return new_state

    def _audio_transcriber_node(self, state: Dict) -> Dict:
        """Audio Transcriber node in the workflow"""
        # Extract agent and task configs
        agent_config = self.agents_config.get('audio_transcriber', {})
        task_config = self.tasks_config.get('process_audio_data', {})
        
        # Prepare inputs
        agent_inputs = {
            "agent_config": f"Role: {agent_config.get('role', '')}\nGoal: {agent_config.get('goal', '')}\nBackstory: {agent_config.get('backstory', '')}",
            "task_config": task_config.get('description', ''),
            "input": f"Process the audio files in: {state.get('audio_dir', 'data/audio')}"
        }
        
        # Run the agent
        result = self.agent_executors["audio_transcriber"].invoke(agent_inputs)
        
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

    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the workflow with the given inputs"""
        if inputs is None:
            inputs = {}
        
        # Execute the workflow
        result = self.workflow.invoke(inputs)
        return result
