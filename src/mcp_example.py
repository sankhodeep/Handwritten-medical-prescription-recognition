#!/usr/bin/env python3
"""
Example script demonstrating how to use the MCP context retrieval module.

This script provides a simple example of how to use the MCP context retrieval
module to retrieve contextual information for prescription processing.

Usage:
    python mcp_example.py
"""

import os
import json
import logging
from retrieval.context_retriever import MCPContextRetriever

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function demonstrating the MCP context retrieval module.
    """
    # Check if the MCP configuration file exists
    config_path = "data/mcp_config.json"
    if not os.path.exists(config_path):
        logger.error(f"MCP configuration file not found at {config_path}")
        return
    
    # Create the context retriever
    logger.info("Creating MCP context retriever")
    retriever = MCPContextRetriever(config_path)
    
    # Example 1: Retrieve medication context
    logger.info("Example 1: Retrieving medication context")
    medication_context = retriever.retrieve_context({
        "chunk_type": "medications",
        "text": "Amoxicillin 500mg 1 tablet 3 times daily for 10 days"
    })
    
    print("\nMedication Context:")
    print(json.dumps(medication_context, indent=2))
    
    # Example 2: Retrieve doctor context
    logger.info("Example 2: Retrieving doctor context")
    doctor_context = retriever.retrieve_context({
        "chunk_type": "doctor_info",
        "text": "Dr. John Smith, Cardiology Department"
    })
    
    print("\nDoctor Context:")
    print(json.dumps(doctor_context, indent=2))
    
    # Example 3: Retrieve symptom context
    logger.info("Example 3: Retrieving symptom context")
    symptom_context = retriever.retrieve_context({
        "chunk_type": "diagnosis",
        "text": "Diagnosis: Acute sinusitis with fever and headache"
    })
    
    print("\nSymptom Context:")
    print(json.dumps(symptom_context, indent=2))
    
    # Example 4: Retrieve context with patient information
    logger.info("Example 4: Retrieving context with patient information")
    combined_context = retriever.retrieve_context({
        "chunk_type": "medications",
        "text": "Amoxicillin 500mg 1 tablet 3 times daily for 10 days",
        "patient_info": "Patient: Jane Doe, Age: 45"
    })
    
    print("\nCombined Context with Patient Information:")
    print(json.dumps(combined_context, indent=2))
    
    logger.info("MCP context retrieval examples completed")

if __name__ == "__main__":
    main()