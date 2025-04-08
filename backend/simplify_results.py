#!/usr/bin/env python3
import os
import json
import glob
import sys

def simplify_results():
    """
    Creates a unified dataset from all result JSON files with only output and agent_type fields.
    """
    try:
        # Use absolute paths for reliability
        base_dir = os.path.dirname(os.path.abspath(__file__))
        result_files_dir = os.path.join(base_dir, "uploads", "processed")
        
        print(f"Looking for files in: {result_files_dir}")
        
        # Find all result JSON files
        result_files = glob.glob(os.path.join(result_files_dir, "*_result.json"))
        print(f"Found {len(result_files)} result files to process.")
        
        # Create the simplified dataset
        simplified_data = {"results": []}
        
        # Process each file
        for result_file in result_files:
            try:
                print(f"Reading file: {result_file}")
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract output and agent_type from extracted_data
                    if "extracted_data" in data:
                        extracted_data = data["extracted_data"]
                        
                        # Only include entries with output and agent_type
                        if "output" in extracted_data and "agent_type" in extracted_data:
                            simplified_data["results"].append({
                                "output": extracted_data["output"],
                                "agent_type": extracted_data["agent_type"]
                            })
                            print(f"  Added entry with output and agent_type")
                        elif "sample_key" in extracted_data:
                            # Handle sample data format
                            simplified_data["results"].append({
                                "output": f"Sample data: {extracted_data.get('sample_key', 'No data')}",
                                "agent_type": data.get("file_type", "unknown")
                            })
                            print(f"  Added entry with sample_key")
                        else:
                            print(f"  No suitable data found in {os.path.basename(result_file)}")
                    else:
                        print(f"  No extracted_data found in {os.path.basename(result_file)}")
                
                print(f"✓ Processed {os.path.basename(result_file)}")
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(result_file)}: {str(e)}")
        
        # Save the simplified dataset
        output_file = os.path.join(base_dir, "simplified_dataset.json")
        with open(output_file, 'w') as f:
            json.dump(simplified_data, f, indent=2)
        
        print(f"Simplified dataset saved to {output_file} with {len(simplified_data['results'])} records.")
        
        # Create a copy in demo_crew directory to ensure it's accessible from the app
        demo_crew_output_file = os.path.join(base_dir, "demo_crew", "processed_data.json")
        with open(demo_crew_output_file, 'w') as f:
            json.dump(simplified_data, f, indent=2)
        
        print(f"Copied dataset to {demo_crew_output_file}")
        
    except Exception as e:
        print(f"An error occurred in the main process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting simplify_results.py script...")
    simplify_results()
    print("Script execution completed.")