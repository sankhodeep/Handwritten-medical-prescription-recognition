#!/usr/bin/env python3
"""
Simplified demo script for the Handwritten Medical Prescription Recognition system.

This script demonstrates the workflow without requiring OpenCV dependencies.
It uses mock implementations for all components.

Usage:
    python demo.py
"""

import os
import logging
import json
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class MockImagePreprocessor:
    """Mock implementation of the image preprocessor."""
    
    def preprocess(self, image_path):
        """Mock preprocessing of an image."""
        logger.info(f"Mock preprocessing image: {image_path}")
        return None, image_path + "_preprocessed"

class MockTextExtractor:
    """Mock implementation of the text extractor."""
    
    def extract_text(self, image_path):
        """Mock extraction of text from an image."""
        logger.info(f"Mock extracting text from: {image_path}")
        return """
        Dr. John Smith
        Medical Center
        123 Health St.
        
        Patient: Jane Doe
        Age: 45
        
        Rx:
        1. Amoxicillin 500mg
           1 tablet 3 times daily for 10 days
        
        2. Ibuprofen 400mg
           1 tablet every 6 hours as needed for pain
        
        Diagnosis: Acute sinusitis
        
        Refer to: Dr. Sarah Johnson, ENT
        
        Follow up in 2 weeks
        
        Signature: J. Smith, MD
        """

class MockTextChunker:
    """Mock implementation of the text chunker."""
    
    def chunk_text(self, raw_text):
        """Mock chunking of text."""
        logger.info("Mock chunking text")
        return {
            "doctor_info": "Dr. John Smith\nMedical Center\n123 Health St.",
            "patient_info": "Patient: Jane Doe\nAge: 45",
            "medications": "1. Amoxicillin 500mg\n   1 tablet 3 times daily for 10 days\n\n2. Ibuprofen 400mg\n   1 tablet every 6 hours as needed for pain",
            "diagnosis": "Diagnosis: Acute sinusitis",
            "referral": "Refer to: Dr. Sarah Johnson, ENT",
            "instructions": "Follow up in 2 weeks",
            "signature": "Signature: J. Smith, MD"
        }

class MockTextRefiner:
    """Mock implementation of the text refiner."""
    
    def refine_chunk(self, chunk_type, chunk_text, context=None):
        """Mock refinement of a text chunk."""
        logger.info(f"Mock refining chunk: {chunk_type}")
        return [chunk_text] * 5  # Return 5 identical versions

class MockContextRetriever:
    """Mock implementation of the context retriever."""
    
    def retrieve_context(self, query):
        """Mock retrieval of context."""
        logger.info(f"Mock retrieving context for: {query.get('chunk_type')}")
        return {
            "medications": [
                {
                    "name": "Amoxicillin",
                    "dosages": ["500mg"],
                    "common_usage": "Bacterial infections"
                }
            ] if query.get("chunk_type") == "medications" else []
        }

class MockSupervisorLLM:
    """Mock implementation of the supervisor LLM."""
    
    def validate(self, transcription, original_image_path):
        """Mock validation of a transcription."""
        logger.info("Mock validating transcription")
        
        # Add validation information
        validated_transcription = transcription.copy()
        validated_transcription["validation"] = {
            "confidence": 0.85,
            "flags": [
                {
                    "type": "warning",
                    "description": "Dosage for Amoxicillin might be unclear",
                    "location": "medications"
                }
            ]
        }
        
        # Add corrected chunks
        validated_transcription["corrected_chunks"] = {}
        for chunk_type, refined_versions in transcription.get("refined_chunks", {}).items():
            if refined_versions:
                validated_transcription["corrected_chunks"][chunk_type] = refined_versions[0]
        
        return validated_transcription, 0.85

class MockOutputGenerator:
    """Mock implementation of the output generator."""
    
    def __init__(self, output_dir="output"):
        """Initialize the mock output generator."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_output(self, transcription, confidence_score, image_path):
        """Mock generation of output."""
        logger.info("Mock generating output")
        
        # Create a simple output
        output = {
            "prescription_id": "demo_prescription_123",
            "confidence_score": confidence_score,
            "transcription": self._format_transcription(transcription),
            "warnings": transcription.get("validation", {}).get("flags", []),
            "original_image": image_path
        }
        
        # Save the output to a JSON file
        output_path = os.path.join(self.output_dir, "demo_prescription_123.json")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Save a text version
        text_path = os.path.join(self.output_dir, "demo_prescription_123.txt")
        with open(text_path, 'w') as f:
            f.write(self._generate_text_output(output))
        
        logger.info(f"Output saved to {output_path} and {text_path}")
        
        return output
    
    def _format_transcription(self, transcription):
        """Format the transcription for output."""
        formatted = {}
        
        # Use corrected chunks if available, otherwise use the best refined version
        corrected_chunks = transcription.get("corrected_chunks", {})
        refined_chunks = transcription.get("refined_chunks", {})
        
        for chunk_type in set(list(corrected_chunks.keys()) + list(refined_chunks.keys())):
            if chunk_type in corrected_chunks:
                formatted[chunk_type] = corrected_chunks[chunk_type]
            elif chunk_type in refined_chunks and refined_chunks[chunk_type]:
                formatted[chunk_type] = refined_chunks[chunk_type][0]
        
        return formatted
    
    def _generate_text_output(self, output):
        """Generate a text output."""
        text = f"PRESCRIPTION TRANSCRIPTION (ID: {output['prescription_id']})\n"
        text += f"Confidence Score: {output['confidence_score']:.2f}\n\n"
        
        # Add each section of the transcription
        for section, content in output["transcription"].items():
            text += f"--- {section.upper()} ---\n"
            text += content + "\n\n"
        
        # Add warnings if any
        if output["warnings"]:
            text += "--- WARNINGS ---\n"
            for warning in output["warnings"]:
                text += f"* {warning['type'].upper()}: {warning['description']} (in {warning['location']})\n"
        
        return text

def main():
    """Main function demonstrating the prescription processing workflow."""
    try:
        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use a sample image path
        image_path = "data/sample_images/README.txt"  # Using README.txt as a placeholder
        
        logger.info(f"Processing prescription image: {image_path}")
        
        # Step 1: Preprocess the image
        logger.info("Step 1: Preprocessing image")
        preprocessor = MockImagePreprocessor()
        _, preprocessed_image_path = preprocessor.preprocess(image_path)
        
        # Step 2: Extract text from the image
        logger.info("Step 2: Extracting text")
        extractor = MockTextExtractor()
        raw_text = extractor.extract_text(preprocessed_image_path)
        
        # Step 3: Chunk and refine the text
        logger.info("Step 3: Chunking and refining text")
        chunker = MockTextChunker()
        refiner = MockTextRefiner()
        
        chunks = chunker.chunk_text(raw_text)
        
        refined_chunks = {}
        for chunk_type, chunk_text in chunks.items():
            logger.info(f"Refining chunk: {chunk_type}")
            
            # Step 4: Retrieve context for the chunk
            logger.info(f"Step 4: Retrieving context for {chunk_type}")
            retriever = MockContextRetriever()
            context = retriever.retrieve_context({
                "chunk_type": chunk_type,
                "text": chunk_text
            })
            
            # Refine the chunk with context
            refined_versions = refiner.refine_chunk(chunk_type, chunk_text, context)
            refined_chunks[chunk_type] = refined_versions
        
        # Create the transcription dictionary
        transcription = {
            "raw_text": raw_text,
            "chunks": chunks,
            "refined_chunks": refined_chunks
        }
        
        # Step 5: Validate the transcription
        logger.info("Step 5: Validating transcription")
        validator = MockSupervisorLLM()
        validated_transcription, confidence_score = validator.validate(transcription, image_path)
        
        # Step 6: Generate output
        logger.info("Step 6: Generating output")
        output_generator = MockOutputGenerator(output_dir)
        output = output_generator.generate_output(validated_transcription, confidence_score, image_path)
        
        # Print the result
        logger.info(f"Prescription processed successfully. Output saved to {output_dir}")
        logger.info(f"Confidence score: {output['confidence_score']:.2f}")
        
        # Print the path to the output files
        for ext in ["json", "txt"]:
            output_path = os.path.join(output_dir, f"{output['prescription_id']}.{ext}")
            if os.path.exists(output_path):
                logger.info(f"Output file: {output_path}")
        
        # Display the text output
        text_path = os.path.join(output_dir, f"{output['prescription_id']}.txt")
        if os.path.exists(text_path):
            logger.info("\nText Output:")
            with open(text_path, 'r') as f:
                print(f.read())
    
    except Exception as e:
        logger.error(f"Error processing prescription: {str(e)}")

if __name__ == "__main__":
    main()