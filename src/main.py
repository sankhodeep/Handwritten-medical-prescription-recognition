#!/usr/bin/env python3
"""
Medical Prescription Processing Automation

This script processes handwritten medical prescriptions and outputs structured,
validated transcriptions. It follows a multi-step approach to ensure accuracy
and contextual understanding of the prescriptions.

Usage:
    python main.py --image <path_to_prescription_image> [options]

Options:
    --image PATH             Path to the prescription image
    --output-dir PATH        Directory to save output files (default: output)
    --output-format FORMAT   Output format: text, html, or both (default: both)
    --use-mock               Use mock implementations instead of actual APIs
    --api-key KEY            Gemini API key (can also be set via GEMINI_API_KEY env var)
    --verbose                Enable verbose logging
    --help                   Show this help message and exit
"""

import os
import sys
import argparse
import logging
import time
from typing import Dict, Any, Optional

# Import modules
from preprocessing.image_processor import ImagePreprocessor
from recognition.text_extractor import get_text_extractor
from refinement.text_refiner import get_text_refiner
from retrieval.context_retriever import get_context_retriever
from validation.prescription_validator import get_supervisor_llm
from output.output_generator import get_output_generator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prescription_processing.log')
    ]
)

logger = logging.getLogger(__name__)

class PrescriptionProcessor:
    """
    Main class for processing prescriptions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the prescription processor.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.use_mock = config.get("use_mock", False)
        self.api_key = config.get("api_key")
        self.output_dir = config.get("output_dir", "output")
        self.output_format = config.get("output_format", "both")
        
        # Initialize components
        self.image_preprocessor = ImagePreprocessor()
        self.text_extractor = get_text_extractor(self.use_mock, self.api_key)
        self.chunker, self.text_refiner = get_text_refiner(self.use_mock, self.api_key)
        self.context_retriever = get_context_retriever(self.use_mock)
        self.supervisor_llm = get_supervisor_llm(self.use_mock, self.api_key)
        self.output_generator = get_output_generator(self.output_format, self.output_dir)
        
        logger.info("Initialized prescription processor")
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Process a prescription image.
        
        Args:
            image_path (str): Path to the prescription image
            
        Returns:
            Dict[str, Any]: Processing result
        """
        try:
            logger.info(f"Processing prescription image: {image_path}")
            start_time = time.time()
            
            # Step 1: Preprocess the image
            logger.info("Step 1: Preprocessing image")
            preprocessed_image, preprocessed_image_path = self.image_preprocessor.preprocess(image_path)
            
            # Step 2: Extract text from the image
            logger.info("Step 2: Extracting text")
            raw_text = self.text_extractor.extract_text(preprocessed_image_path)
            
            # Step 3: Chunk and refine the text
            logger.info("Step 3: Chunking and refining text")
            chunks = self.chunker.chunk_text(raw_text)
            
            refined_chunks = {}
            for chunk_type, chunk_text in chunks.items():
                logger.info(f"Refining chunk: {chunk_type}")
                
                # Step 4: Retrieve context for the chunk
                logger.info(f"Step 4: Retrieving context for {chunk_type}")
                context = self.context_retriever.retrieve_context({
                    "chunk_type": chunk_type,
                    "text": chunk_text
                })
                
                # Refine the chunk with context
                refined_versions = self.text_refiner.refine_chunk(chunk_type, chunk_text, context)
                refined_chunks[chunk_type] = refined_versions
            
            # Create the transcription dictionary
            transcription = {
                "raw_text": raw_text,
                "chunks": chunks,
                "refined_chunks": refined_chunks
            }
            
            # Step 5: Validate the transcription
            logger.info("Step 5: Validating transcription")
            validated_transcription, confidence_score = self.supervisor_llm.validate(transcription, image_path)
            
            # If confidence score is below threshold, restart the process
            if confidence_score < 0.8 and not self.use_mock:
                logger.warning(f"Confidence score {confidence_score} is below threshold. Restarting process.")
                
                # For simplicity, we'll just run the validation again
                # In a real implementation, you might want to adjust parameters or try different approaches
                logger.info("Revalidating transcription")
                validated_transcription, confidence_score = self.supervisor_llm.validate(transcription, image_path)
            
            # Step 6: Generate output
            logger.info("Step 6: Generating output")
            output = self.output_generator.generate_output(validated_transcription, confidence_score, image_path)
            
            end_time = time.time()
            logger.info(f"Prescription processing completed in {end_time - start_time:.2f} seconds")
            
            return output
        
        except Exception as e:
            logger.error(f"Error processing prescription: {str(e)}")
            raise

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Medical Prescription Processing Automation")
    
    parser.add_argument("--image", required=True, help="Path to the prescription image")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--output-format", default="both", choices=["text", "html", "both"], help="Output format")
    parser.add_argument("--use-mock", action="store_true", help="Use mock implementations instead of actual APIs")
    parser.add_argument("--api-key", help="Gemini API key (can also be set via GEMINI_API_KEY env var)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def main() -> None:
    """
    Main function.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if the image file exists
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        sys.exit(1)
    
    # Create the configuration dictionary
    config = {
        "output_dir": args.output_dir,
        "output_format": args.output_format,
        "use_mock": args.use_mock,
        "api_key": args.api_key
    }
    
    # Create the prescription processor
    processor = PrescriptionProcessor(config)
    
    try:
        # Process the prescription
        result = processor.process(args.image)
        
        # Print the result
        logger.info(f"Prescription processed successfully. Output saved to {args.output_dir}")
        logger.info(f"Confidence score: {result['confidence_score']:.2f}")
        
        # Print the path to the output files
        for ext in ["json", "txt", "html"]:
            output_path = os.path.join(args.output_dir, f"{result['prescription_id']}.{ext}")
            if os.path.exists(output_path):
                logger.info(f"Output file: {output_path}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()