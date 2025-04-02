import os
import logging
import base64
import requests
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GeminiVisionTextExtractor:
    """
    Class for extracting text from prescription images using the Gemini Vision API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini Vision text extractor.
        
        Args:
            api_key (str, optional): Gemini API key. If not provided, it will try to get it from
                                     the GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No Gemini API key provided. Using placeholder for development.")
            self.api_key = "PLACEHOLDER_API_KEY"
        
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
        logger.info("Initialized Gemini Vision text extractor")
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode the image to base64.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from the prescription image using Gemini Vision API.
        
        Args:
            image_path (str): Path to the preprocessed image file
            
        Returns:
            str: Extracted raw text from the image
        """
        try:
            # Encode the image
            encoded_image = self._encode_image(image_path)
            
            # Prepare the request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": "Extract the exact text from the prescription image without modifying or correcting it. Preserve every word as written."
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": encoded_image
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Set up the request headers
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            # Make the API request
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=payload
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract the text from the response
                extracted_text = self._parse_gemini_response(response_data)
                logger.info("Successfully extracted text from image")
                return extracted_text
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                # For development/testing, return a placeholder if the API call fails
                if self.api_key == "PLACEHOLDER_API_KEY":
                    logger.warning("Using placeholder text since API key is not set")
                    return "This is placeholder extracted text for development purposes."
                raise Exception(f"API request failed: {response.text}")
        
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def _parse_gemini_response(self, response_data: Dict[str, Any]) -> str:
        """
        Parse the response from the Gemini API to extract the text.
        
        Args:
            response_data (Dict[str, Any]): Response data from the Gemini API
            
        Returns:
            str: Extracted text
        """
        try:
            # Extract text from the response
            # The exact structure may vary based on the Gemini API response format
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    for part in parts:
                        if "text" in part:
                            return part["text"].strip()
            
            # If we couldn't extract the text using the expected structure
            logger.warning("Could not parse Gemini API response, returning raw response")
            return json.dumps(response_data)
        
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            raise

class MockVisionTextExtractor:
    """
    Mock implementation of the text extractor for testing without API access.
    """
    
    def __init__(self):
        """Initialize the mock text extractor."""
        logger.info("Initialized mock vision text extractor")
    
    def extract_text(self, image_path: str) -> str:
        """
        Mock extraction of text from the prescription image.
        
        Args:
            image_path (str): Path to the preprocessed image file
            
        Returns:
            str: Mock extracted text
        """
        logger.info(f"Mock extracting text from {image_path}")
        
        # Return mock prescription text for testing
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

def get_text_extractor(use_mock: bool = False, api_key: Optional[str] = None):
    """
    Factory function to get the appropriate text extractor.
    
    Args:
        use_mock (bool): Whether to use the mock extractor
        api_key (str, optional): Gemini API key
        
    Returns:
        TextExtractor: An instance of a text extractor
    """
    if use_mock:
        return MockVisionTextExtractor()
    else:
        return GeminiVisionTextExtractor(api_key)