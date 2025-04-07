# Autonomous Document Intelligence

Welcome to the Autonomous Document Intelligence project, a mystery-themed multi-agent system powered by [crewAI](https://crewai.com). This project is designed to help you process unstructured documents using a team of specialized agents that collaborate to extract meaningful information, analyze content, and transform raw data into structured datasets.

## Project Overview

The Autonomous Document Intelligence system addresses the challenges enterprises face in transforming unstructured and semi-structured documents into structured datasets. It employs a team of specialized agents - a Detective, Forensic Analyst, Researcher, and Profiler - that work together to analyze multimedia evidence, research background information, create psychological profiles, and ultimately solve document-related mysteries.

Key features:
- Natural language-driven data curation
- Multimodal content processing (text, images, audio, video)
- Advanced document analysis and information extraction
- Dynamic schema adaptation based on user needs
- Continuous learning from interactions

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

```bash
crewai install
```

### Configuration

**Add your `OPENAI_API_KEY` into the `.env` file**

The project is structured around:
- `src/demo_crew/config/agents.yaml`: Defines the mystery-solving agents
- `src/demo_crew/config/tasks.yaml`: Defines document processing tasks
- `src/demo_crew/crew.py`: Configures the multi-agent system with tools and logic
- `src/demo_crew/tools/`: Contains specialized tools for processing different content types

## Running the Project

To kickstart your crew of document intelligence agents and begin processing, run:

```bash
$ crewai run
```

This command initializes the Mystery Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example will analyze a mystery case and create a comprehensive `mystery_solution.md` report in the root folder.

## Understanding Your Crew

The Mystery Crew is composed of multiple AI agents:

1. **Detective**: Lead investigator who coordinates the investigation and solves the document mystery
2. **Forensic Analyst**: Specializes in analyzing multimedia evidence (images, audio, video)
3. **Researcher**: Gathers background information and contextual data
4. **Profiler**: Creates psychological profiles and analyzes behavioral patterns

These agents collaborate on tasks defined in `config/tasks.yaml`, using specialized tools for different content types.

## Support

For support, questions, or feedback:
- Visit crewAI [documentation](https://docs.crewai.com)
- Reach out through the [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join the Discord](https://discord.com/invite/X4JWnZnxPb)

Unlock the power of intelligent document processing with our mystery-solving agents!
