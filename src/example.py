#!/usr/bin/env python3
"""
Example script demonstrating how to use the prescription processing system.

This script provides a simple example of how to use the prescription processing
system with a sample prescription image.

Usage:
    python example.py
"""

import os
import logging
from main import PrescriptionProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function demonstrating the prescription processing system.
    """
    # Create sample directories if they don't exist
    os.makedirs("data/sample_images", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Check if we have a sample image
    sample_image_path = "data/sample_images/sample_prescription.jpg"
    
    if not os.path.exists(sample_image_path):
        logger.warning(f"Sample image not found at {sample_image_path}")
        logger.info("Please add a sample prescription image to data/sample_images/sample_prescription.jpg")
        logger.info("Alternatively, you can specify a different image path in this script.")
        return
    
    # Configuration for the processor
    config = {
        "output_dir": "output",
        "output_format": "both",
        "use_mock": True,  # Use mock implementations for demonstration
        "api_key": None    # No API key needed for mock implementations
    }
    
    # Create the prescription processor
    processor = PrescriptionProcessor(config)
    
    try:
        # Process the prescription
        logger.info(f"Processing sample prescription image: {sample_image_path}")
        result = processor.process(sample_image_path)
        
        # Print the result
        logger.info("Prescription processed successfully!")
        logger.info(f"Confidence score: {result['confidence_score']:.2f}")
        
        # Print the path to the output files
        for ext in ["json", "txt", "html"]:
            output_path = os.path.join(config["output_dir"], f"{result['prescription_id']}.{ext}")
            if os.path.exists(output_path):
                logger.info(f"Output file: {output_path}")
        
        # Print instructions for viewing the HTML output
        html_path = os.path.join(config["output_dir"], f"{result['prescription_id']}.html")
        if os.path.exists(html_path):
            logger.info(f"To view the HTML output, open {html_path} in a web browser.")
    
    except Exception as e:
        logger.error(f"Error processing prescription: {str(e)}")

if __name__ == "__main__":
    main()