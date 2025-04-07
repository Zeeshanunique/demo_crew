from crewai import Agent, Crew, Task
from typing import List
import yaml
import os

from .tools.tool_factory import (
    get_document_rag_tool,
    get_web_search_tool,
    get_ocr_tool,
    get_image_analysis_tool,
    get_whisper_transcription_tool,
    get_video_tool
)

class DocumentCrew:
    """Document Intelligence crew for processing various file types"""

    def __init__(self):
        # Load configuration files
        self.agents_config = self._load_config('config/agents.yaml')
        self.tasks_config = self._load_config('config/tasks.yaml')
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()

    def _load_config(self, path):
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

    def _create_agents(self) -> List[Agent]:
        """Create the agents for the crew"""
        text_specialist = Agent(
            role=self.agents_config['text_data_specialist']['role'],
            goal=self.agents_config['text_data_specialist']['goal'],
            backstory=self.agents_config['text_data_specialist']['backstory'],
            tools=[get_document_rag_tool(), get_web_search_tool()],
            verbose=True
        )

        image_analyst = Agent(
            role=self.agents_config['image_ocr_analyst']['role'],
            goal=self.agents_config['image_ocr_analyst']['goal'],
            backstory=self.agents_config['image_ocr_analyst']['backstory'],
            tools=[get_ocr_tool(), get_image_analysis_tool()],
            verbose=True
        )

        video_specialist = Agent(
            role=self.agents_config['video_transcriber']['role'],
            goal=self.agents_config['video_transcriber']['goal'],
            backstory=self.agents_config['video_transcriber']['backstory'],
            tools=[get_video_tool()],
            verbose=True
        )

        audio_specialist = Agent(
            role=self.agents_config['audio_transcriber']['role'],
            goal=self.agents_config['audio_transcriber']['goal'],
            backstory=self.agents_config['audio_transcriber']['backstory'],
            tools=[get_whisper_transcription_tool()],
            verbose=True
        )

        return [text_specialist, image_analyst, video_specialist, audio_specialist]

    def _create_tasks(self) -> List[Task]:
        """Create the tasks for the crew"""
        process_text = Task(
            description=self.tasks_config['process_text_data']['description'],
            expected_output=self.tasks_config['process_text_data']['expected_output'],
            agent=self.agents[0],  # text_data_specialist
        )

        process_image = Task(
            description=self.tasks_config['process_image_data']['description'],
            expected_output=self.tasks_config['process_image_data']['expected_output'],
            agent=self.agents[1],  # image_ocr_analyst
        )

        process_video = Task(
            description=self.tasks_config['process_video_data']['description'],
            expected_output=self.tasks_config['process_video_data']['expected_output'],
            agent=self.agents[2],  # video_transcriber
        )

        process_audio = Task(
            description=self.tasks_config['process_audio_data']['description'],
            expected_output=self.tasks_config['process_audio_data']['expected_output'],
            agent=self.agents[3],  # audio_transcriber
            output_file='processed_data.json'
        )

        return [process_text, process_image, process_video, process_audio]

    def create_crew(self) -> Crew:
        """Creates the Document Intelligence crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
