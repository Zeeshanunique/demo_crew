"""
LangSmith integration module for monitoring and tracing LangChain applications.
This module provides utilities for setting up LangSmith tracing and evaluation.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

# Import LangSmith
try:
    from langsmith import Client
    from langsmith.run_trees import RunTree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("langsmith_integration")

def setup_langsmith() -> Optional[Client]:
    """
    Set up LangSmith for tracing and monitoring LangChain applications.
    
    Returns:
        Optional[Client]: LangSmith client if successfully initialized, None otherwise
    """
    # Load environment variables
    load_dotenv()
    
    # Check if LangSmith is available
    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith package is not installed. Install with: pip install langsmith")
        return None
    
    # Check for LangSmith environment variables
    langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")
    langchain_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    
    if not langchain_api_key:
        logger.info("LANGCHAIN_API_KEY not set. LangSmith tracing will be disabled.")
        return None
        
    if not langchain_tracing:
        logger.info("LANGCHAIN_TRACING_V2 is not set to 'true'. LangSmith tracing will be disabled.")
        return None
    
    # Initialize LangSmith client
    try:
        project_name = os.environ.get("LANGCHAIN_PROJECT", "demo_crew")
        langsmith_client = Client(
            api_key=langchain_api_key,
            api_url=os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        )
        
        # Test connection to LangSmith
        try:
            projects = list(langsmith_client.list_projects())
            logger.info(f"LangSmith connection successful! Found {len(projects)} projects.")
            
            # Check if project exists, create if not
            if not any(p.name == project_name for p in projects):
                logger.info(f"Creating new LangSmith project: {project_name}")
                langsmith_client.create_project(project_name)
            
            logger.info(f"LangSmith tracing enabled for project: {project_name}")
        except Exception as e:
            logger.warning(f"LangSmith connection test failed: {str(e)}")
            return None
            
        return langsmith_client
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {str(e)}")
        return None

def create_evaluation_dataset(client: Client, 
                             runs: List[str], 
                             dataset_name: str, 
                             description: str = None) -> str:
    """
    Create a LangSmith evaluation dataset from a list of run IDs.
    
    Args:
        client: LangSmith client
        runs: List of run IDs to include in the dataset
        dataset_name: Name of the dataset to create
        description: Optional description of the dataset
        
    Returns:
        str: ID of the created dataset
    """
    if not client:
        logger.warning("LangSmith client not initialized. Cannot create evaluation dataset.")
        return None
    
    try:
        # Create the dataset
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description or f"Evaluation dataset created on {dataset_name}"
        )
        
        # Add runs to the dataset
        for run_id in runs:
            run_tree = RunTree.from_id(run_id, client=client)
            
            # Extract inputs and outputs
            inputs = run_tree.inputs
            outputs = run_tree.outputs or {}
            
            # Add example to dataset
            client.create_example(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id
            )
        
        logger.info(f"Created evaluation dataset '{dataset_name}' with {len(runs)} examples")
        return dataset.id
    except Exception as e:
        logger.error(f"Failed to create evaluation dataset: {str(e)}")
        return None

def track_agent_performance(client: Client, 
                           run_id: str,
                           agent_type: str,
                           metrics: Dict[str, Union[float, int, str]]) -> bool:
    """
    Track agent performance metrics in LangSmith.
    
    Args:
        client: LangSmith client
        run_id: Run ID to associate metrics with
        agent_type: Type of agent (text, image, audio, video)
        metrics: Dictionary of metrics to track
        
    Returns:
        bool: True if metrics were tracked successfully, False otherwise
    """
    if not client:
        logger.warning("LangSmith client not initialized. Cannot track metrics.")
        return False
    
    try:
        # Add metrics to the run
        for key, value in metrics.items():
            client.update_run_metadata(
                run_id=run_id,
                key=f"{agent_type}_{key}",
                value=value
            )
        
        logger.info(f"Added {len(metrics)} metrics to run {run_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to track metrics: {str(e)}")
        return False