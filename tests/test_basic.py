#!/usr/bin/env python3
"""
Basic tests for the prescription processing system.

This script contains basic tests to verify that the prescription processing
system is functioning correctly.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.image_processor import ImagePreprocessor
from src.recognition.text_extractor import MockVisionTextExtractor
from src.refinement.text_refiner import TextChunker, MockTextRefiner
from src.retrieval.context_retriever import MockContextRetriever
from src.validation.prescription_validator import MockSupervisorLLM
from src.output.output_generator import OutputGenerator
from src.main import PrescriptionProcessor

class TestPreprocessing(unittest.TestCase):
    """Test the image preprocessing module."""
    
    def setUp(self):
        """Set up the test."""
        self.preprocessor = ImagePreprocessor()
    
    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_load_image(self, mock_imwrite, mock_imread):
        """Test loading an image."""
        # Mock the image
        mock_image = MagicMock()
        mock_imread.return_value = mock_image
        
        # Test loading the image
        image = self.preprocessor.load_image("test.jpg")
        
        # Check that the image was loaded
        mock_imread.assert_called_once_with("test.jpg")
        self.assertEqual(image, mock_image)

class TestRecognition(unittest.TestCase):
    """Test the text recognition module."""
    
    def setUp(self):
        """Set up the test."""
        self.extractor = MockVisionTextExtractor()
    
    def test_extract_text(self):
        """Test extracting text from an image."""
        # Test extracting text
        text = self.extractor.extract_text("test.jpg")
        
        # Check that text was extracted
        self.assertIsNotNone(text)
        self.assertIn("Dr. John Smith", text)

class TestRefinement(unittest.TestCase):
    """Test the text refinement module."""
    
    def setUp(self):
        """Set up the test."""
        self.chunker = TextChunker()
        self.refiner = MockTextRefiner()
    
    def test_chunk_text(self):
        """Test chunking text."""
        # Sample text
        text = """
        Dr. John Smith
        Medical Center
        
        Patient: Jane Doe
        Age: 45
        
        Rx:
        1. Amoxicillin 500mg
           1 tablet 3 times daily for 10 days
        
        Diagnosis: Acute sinusitis
        """
        
        # Test chunking the text
        chunks = self.chunker.chunk_text(text)
        
        # Check that the text was chunked
        self.assertIn("doctor_info", chunks)
        self.assertIn("patient_info", chunks)
        self.assertIn("medications", chunks)
        self.assertIn("diagnosis", chunks)
    
    def test_refine_chunk(self):
        """Test refining a chunk."""
        # Test refining a chunk
        refined = self.refiner.refine_chunk("medications", "1. Amoxicillin 500mg")
        
        # Check that the chunk was refined
        self.assertEqual(len(refined), 5)  # Should have 5 versions

class TestRetrieval(unittest.TestCase):
    """Test the contextual retrieval module."""
    
    def setUp(self):
        """Set up the test."""
        self.retriever = MockContextRetriever()
    
    def test_retrieve_context(self):
        """Test retrieving context."""
        # Test retrieving context
        context = self.retriever.retrieve_context({
            "chunk_type": "medications",
            "text": "Amoxicillin 500mg"
        })
        
        # Check that context was retrieved
        self.assertIn("medications", context)
        self.assertTrue(len(context["medications"]) > 0)

class TestValidation(unittest.TestCase):
    """Test the validation module."""
    
    def setUp(self):
        """Set up the test."""
        self.validator = MockSupervisorLLM()
    
    def test_validate(self):
        """Test validating a transcription."""
        # Sample transcription
        transcription = {
            "raw_text": "Sample text",
            "chunks": {
                "medications": "Amoxicillin 500mg"
            },
            "refined_chunks": {
                "medications": ["Amoxicillin 500mg", "Amoxicillin 500 mg"]
            }
        }
        
        # Test validating the transcription
        validated, score = self.validator.validate(transcription, "test.jpg")
        
        # Check that the transcription was validated
        self.assertIsNotNone(validated)
        self.assertGreater(score, 0)
        self.assertIn("validation", validated)

class TestOutput(unittest.TestCase):
    """Test the output generation module."""
    
    def setUp(self):
        """Set up the test."""
        # Create a temporary output directory
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.generator = OutputGenerator(self.output_dir)
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove the temporary output directory
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    @patch('json.dump')
    @patch('builtins.open', new_callable=MagicMock)
    def test_generate_output(self, mock_open, mock_json_dump):
        """Test generating output."""
        # Sample transcription
        transcription = {
            "raw_text": "Sample text",
            "chunks": {
                "medications": "Amoxicillin 500mg"
            },
            "refined_chunks": {
                "medications": ["Amoxicillin 500mg", "Amoxicillin 500 mg"]
            },
            "validation": {
                "confidence": 0.85,
                "flags": []
            }
        }
        
        # Test generating output
        output = self.generator.generate_output(transcription, 0.85, "test.jpg")
        
        # Check that output was generated
        self.assertIsNotNone(output)
        self.assertIn("prescription_id", output)
        self.assertIn("confidence_score", output)
        self.assertIn("transcription", output)

class TestPrescriptionProcessor(unittest.TestCase):
    """Test the main prescription processor."""
    
    def setUp(self):
        """Set up the test."""
        # Configuration for the processor
        self.config = {
            "output_dir": "test_output",
            "output_format": "text",
            "use_mock": True,
            "api_key": None
        }
        
        # Create a temporary output directory
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        self.processor = PrescriptionProcessor(self.config)
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove the temporary output directory
        import shutil
        if os.path.exists(self.config["output_dir"]):
            shutil.rmtree(self.config["output_dir"])
    
    @patch('src.preprocessing.image_processor.ImagePreprocessor.preprocess')
    def test_process(self, mock_preprocess):
        """Test processing a prescription."""
        # Mock the preprocessing
        mock_preprocess.return_value = (MagicMock(), "test_preprocessed.jpg")
        
        # Create a temporary test image
        test_image = os.path.join(self.config["output_dir"], "test.jpg")
        with open(test_image, 'w') as f:
            f.write("dummy image content")
        
        # Test processing the prescription
        result = self.processor.process(test_image)
        
        # Check that the prescription was processed
        self.assertIsNotNone(result)
        self.assertIn("prescription_id", result)
        self.assertIn("confidence_score", result)
        self.assertIn("transcription", result)

if __name__ == "__main__":
    unittest.main()