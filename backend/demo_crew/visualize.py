"""
Visualization utilities for the LangGraph workflow
"""

import os
import json
from typing import Dict, Any, Optional
from demo_crew.document_graph import DocumentGraph

def visualize_graph(output_path: str = "graph.html") -> str:
    """
    Create a visualization of the LangGraph workflow
    
    Args:
        output_path: Path where the HTML visualization will be saved
        
    Returns:
        The path to the generated visualization file
    """
    try:
        # Initialize the document graph
        document_graph = DocumentGraph()
        
        # Get the workflow
        workflow = document_graph.workflow
        
        # Generate the visualization to HTML
        from langgraph.graph import StateGraph, END
        
        # The compiled workflow doesn't have a direct visualization method
        # We need to recreate a visualization-friendly version
        viz_graph = StateGraph(Dict)
        
        # Add nodes
        viz_graph.add_node("text_data_specialist", document_graph._text_data_specialist_node)
        viz_graph.add_node("image_ocr_analyst", document_graph._image_ocr_analyst_node)
        viz_graph.add_node("video_transcriber", document_graph._video_transcriber_node)
        viz_graph.add_node("audio_transcriber", document_graph._audio_transcriber_node)
        viz_graph.add_node("compile_results", document_graph._compile_results)
        
        # Define the workflow
        viz_graph.add_edge("text_data_specialist", "image_ocr_analyst")
        viz_graph.add_edge("image_ocr_analyst", "video_transcriber")
        viz_graph.add_edge("video_transcriber", "audio_transcriber")
        viz_graph.add_edge("audio_transcriber", "compile_results")
        viz_graph.add_edge("compile_results", END)
        
        # Set the entry point
        viz_graph.set_entry_point("text_data_specialist")
        
        # Get visualization and save it
        from IPython.display import HTML
        viz_html = viz_graph.get_graph().to_html()
        
        with open(output_path, "w") as f:
            f.write(viz_html)
        
        return output_path
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return ""

def save_workflow_results(results: Dict[str, Any], output_path: str = "workflow_results.json") -> None:
    """
    Save the workflow results to a formatted JSON file
    
    Args:
        results: The results dictionary from the workflow run
        output_path: Path where the JSON results will be saved
    """
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def analyze_workflow_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the workflow performance from results
    
    Args:
        results: The results dictionary from the workflow run
        
    Returns:
        Dictionary with performance metrics
    """
    metrics = {
        "steps_executed": 0,
        "total_tokens": 0,
        "processing_summary": {}
    }
    
    # Count steps
    if "text_processing_result" in results:
        metrics["steps_executed"] += 1
        metrics["processing_summary"]["text_processing"] = len(str(results["text_processing_result"]))
    
    if "image_processing_result" in results:
        metrics["steps_executed"] += 1
        metrics["processing_summary"]["image_processing"] = len(str(results["image_processing_result"]))
    
    if "video_processing_result" in results:
        metrics["steps_executed"] += 1
        metrics["processing_summary"]["video_processing"] = len(str(results["video_processing_result"]))
    
    if "audio_processing_result" in results:
        metrics["steps_executed"] += 1
        metrics["processing_summary"]["audio_processing"] = len(str(results["audio_processing_result"]))
    
    if "final_output" in results:
        metrics["steps_executed"] += 1
        
    # Estimate tokens (rough calculation)
    for key, value in results.items():
        if isinstance(value, str):
            metrics["total_tokens"] += len(value.split())
    
    return metrics

if __name__ == "__main__":
    # Example usage
    viz_path = visualize_graph()
    print(f"Visualization saved to: {viz_path}")