# Autonomous Document Intelligence

Welcome to the Autonomous Document Intelligence project, a document processing system powered by [LangGraph](https://langchain.readthedocs.io/projects/langgraph) and [LangChain](https://langchain.com). This project is designed to help you process unstructured documents using a team of specialized agents that collaborate to extract meaningful information, analyze content, and transform raw data into structured datasets.

## Project Overview

The Autonomous Document Intelligence system addresses the challenges enterprises face in transforming unstructured and semi-structured documents into structured datasets. It employs a team of specialized agents - Text Data Specialist, OCR Image Analyst, Video Transcriber, and Audio Transcriber - that work together in a LangGraph workflow to analyze multimedia evidence, extract information, and compile results into structured data.

Key features:
- Natural language-driven data curation
- Multimodal content processing (text, images, audio, video)
- Advanced document analysis and information extraction
- LangGraph workflow orchestration
- Visualization of agent interactions

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system.

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/document-intelligence.git
cd document-intelligence

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

**Add your `OPENAI_API_KEY` into the `.env` file**

The project is structured around:
- `demo_crew/config/agents.yaml`: Defines the specialized agents
- `demo_crew/config/tasks.yaml`: Defines document processing tasks
- `demo_crew/crew.py`: Configures the LangGraph workflow with tools and logic
- `demo_crew/tools/`: Contains specialized tools for processing different content types

## Running the Project

To run the document intelligence workflow:

```bash
$ python -m demo_crew.main
```

This command initializes the LangGraph workflow, executing the agents in sequence to process documents, images, audio, and video files.

For a visual representation of the workflow:

```bash
$ python -m demo_crew.visualize
```

This will create an HTML visualization of the LangGraph workflow that you can open in your browser.

## Understanding the LangGraph Workflow

The document intelligence workflow consists of multiple nodes in a LangGraph state machine:

1. **Text Data Specialist**: Processes text documents using RAG and web search tools
2. **OCR Image Analyst**: Analyzes images and extracts text with OCR
3. **Video Transcriber**: Processes video files to extract audio and visual information
4. **Audio Transcriber**: Converts audio to text with metadata and analysis
5. **Results Compiler**: Assembles results from all agents into structured output

The workflow executes these nodes in sequence, with each node enhancing the shared state with its processing results.

## Tools and Technologies

This project leverages several key technologies:

- **LangGraph**: For workflow orchestration and agent coordination
- **LangChain**: For agent creation, tool definition, and LLM interactions
- **OpenAI**: For language model capabilities
- **PyTesseract**: For OCR functionality
- **OpenCV**: For image and video processing
- **Whisper**: For audio transcription

## Extending the System

To add a new agent to the workflow:

1. Define a new agent configuration in `config/agents.yaml`
2. Create tools for the agent in the `tools` directory
3. Add a new node function in `crew.py` for the agent's processing step
4. Update the workflow graph in `_create_workflow()` to include the new node

## Support

For support, questions, or feedback, please open an issue in the GitHub repository.

Happy document processing!
