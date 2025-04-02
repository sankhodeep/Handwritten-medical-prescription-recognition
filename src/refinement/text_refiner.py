import os
import logging
import re
import requests
import json
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Class for splitting raw prescription text into logical chunks.
    """
    
    def __init__(self):
        """Initialize the text chunker."""
        logger.info("Initialized text chunker")
    
    def chunk_text(self, raw_text: str) -> Dict[str, str]:
        """
        Split the raw text into logical chunks.
        
        Args:
            raw_text (str): Raw extracted text from the prescription
            
        Returns:
            Dict[str, str]: Dictionary of chunks with keys like 'doctor_info', 
                           'patient_info', 'medications', 'instructions', etc.
        """
        try:
            # Initialize chunks dictionary
            chunks = {
                'doctor_info': '',
                'patient_info': '',
                'medications': '',
                'diagnosis': '',
                'instructions': '',
                'referral': '',
                'signature': '',
                'other': ''
            }
            
            # Split text into lines and remove empty lines
            lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
            
            # Process each line to determine which chunk it belongs to
            current_chunk = 'other'
            
            for line in lines:
                # Check for doctor information (usually at the top)
                if re.search(r'dr\.?|doctor|medical center|clinic|hospital', line, re.IGNORECASE) and not chunks['doctor_info']:
                    current_chunk = 'doctor_info'
                
                # Check for patient information
                elif re.search(r'patient|name|age|dob|date of birth', line, re.IGNORECASE):
                    current_chunk = 'patient_info'
                
                # Check for medication information
                elif re.search(r'rx|medication|tablet|capsule|mg|ml|take|dose', line, re.IGNORECASE):
                    current_chunk = 'medications'
                
                # Check for diagnosis
                elif re.search(r'diagnosis|assessment|condition', line, re.IGNORECASE):
                    current_chunk = 'diagnosis'
                
                # Check for instructions
                elif re.search(r'instruction|direction|follow|advice', line, re.IGNORECASE):
                    current_chunk = 'instructions'
                
                # Check for referral
                elif re.search(r'refer|consult|specialist', line, re.IGNORECASE):
                    current_chunk = 'referral'
                
                # Check for signature
                elif re.search(r'sign|signature|md|doctor', line, re.IGNORECASE) and line.strip().startswith(('Sign', 'Signature')):
                    current_chunk = 'signature'
                
                # Add the line to the current chunk
                chunks[current_chunk] += line + '\n'
            
            # Clean up chunks (remove trailing newlines and empty chunks)
            for key in chunks:
                chunks[key] = chunks[key].strip()
            
            # If medications chunk is empty but there are numbered items in 'other', 
            # they might be medications
            if not chunks['medications'] and re.search(r'\d+\.\s', chunks['other']):
                chunks['medications'] = chunks['other']
                chunks['other'] = ''
            
            logger.info("Successfully chunked text into logical sections")
            return {k: v for k, v in chunks.items() if v}  # Return only non-empty chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            # Return the raw text as a single chunk if chunking fails
            return {'full_text': raw_text}

class GeminiTextRefiner:
    """
    Class for refining text chunks using the Gemini API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini text refiner.
        
        Args:
            api_key (str, optional): Gemini API key. If not provided, it will try to get it from
                                     the GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No Gemini API key provided. Using placeholder for development.")
            self.api_key = "PLACEHOLDER_API_KEY"
        
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        logger.info("Initialized Gemini text refiner")
    
    def refine_chunk(self, chunk_type: str, chunk_text: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Refine a text chunk using the Gemini API.
        
        Args:
            chunk_type (str): Type of the chunk (e.g., 'medications', 'doctor_info')
            chunk_text (str): Text content of the chunk
            context (Dict[str, Any], optional): Additional context for refinement
            
        Returns:
            List[str]: List of 5 refined versions of the chunk
        """
        try:
            # Prepare the prompt based on chunk type
            prompt = self._create_prompt_for_chunk(chunk_type, chunk_text, context)
            
            # Prepare the request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 1024,
                    "stopSequences": []
                }
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
                
                # Extract the refined versions from the response
                refined_versions = self._parse_gemini_response(response_data)
                logger.info(f"Successfully refined {chunk_type} chunk")
                return refined_versions
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                # For development/testing, return placeholders if the API call fails
                if self.api_key == "PLACEHOLDER_API_KEY":
                    logger.warning("Using placeholder refined versions since API key is not set")
                    return [f"Refined version {i+1} of {chunk_text}" for i in range(5)]
                raise Exception(f"API request failed: {response.text}")
        
        except Exception as e:
            logger.error(f"Error refining chunk: {str(e)}")
            # Return the original chunk text as the only version if refinement fails
            return [chunk_text]
    
    def _create_prompt_for_chunk(self, chunk_type: str, chunk_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for the Gemini API based on the chunk type.
        
        Args:
            chunk_type (str): Type of the chunk
            chunk_text (str): Text content of the chunk
            context (Dict[str, Any], optional): Additional context for refinement
            
        Returns:
            str: Prompt for the Gemini API
        """
        # Base prompt template
        base_prompt = f"""
        You are an expert medical transcriptionist specializing in handwritten prescription interpretation.
        
        I have a handwritten prescription that has been OCR-processed, but it may contain errors due to handwriting recognition challenges.
        
        Below is the extracted text for the {chunk_type.replace('_', ' ')} section:
        
        ```
        {chunk_text}
        ```
        
        Please generate 5 different refined versions of this text, correcting potential OCR errors while preserving the original meaning.
        Each version should be a complete, coherent refinement of the entire text.
        
        Format your response as a JSON array with 5 strings, each representing a different refined version.
        Example: ["Version 1", "Version 2", "Version 3", "Version 4", "Version 5"]
        
        Only provide the JSON array, no additional text.
        """
        
        # Add context-specific instructions based on chunk type
        if chunk_type == 'medications':
            base_prompt += """
            For medications:
            - Ensure medication names are spelled correctly
            - Verify dosages are in standard formats (mg, ml, etc.)
            - Check that administration instructions are clear
            - Ensure frequency and duration are properly formatted
            """
        elif chunk_type == 'doctor_info':
            base_prompt += """
            For doctor information:
            - Ensure doctor names are properly formatted
            - Verify medical institution names
            - Check for proper formatting of addresses and contact information
            """
        elif chunk_type == 'patient_info':
            base_prompt += """
            For patient information:
            - Ensure patient names are properly formatted
            - Verify age, date of birth, or other demographic information
            - Check for any patient ID numbers or other identifiers
            """
        
        # Add any additional context if provided
        if context:
            context_str = "\n\nAdditional context:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
            base_prompt += context_str
        
        return base_prompt
    
    def _parse_gemini_response(self, response_data: Dict[str, Any]) -> List[str]:
        """
        Parse the response from the Gemini API to extract the refined versions.
        
        Args:
            response_data (Dict[str, Any]): Response data from the Gemini API
            
        Returns:
            List[str]: List of refined versions
        """
        try:
            # Extract text from the response
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    for part in parts:
                        if "text" in part:
                            # Try to parse the response as JSON
                            try:
                                refined_versions = json.loads(part["text"].strip())
                                if isinstance(refined_versions, list) and len(refined_versions) > 0:
                                    # Ensure we have exactly 5 versions
                                    if len(refined_versions) < 5:
                                        # Duplicate the last version if we have fewer than 5
                                        refined_versions.extend([refined_versions[-1]] * (5 - len(refined_versions)))
                                    return refined_versions[:5]
                            except json.JSONDecodeError:
                                # If JSON parsing fails, try to extract versions using regex
                                versions = re.findall(r'"([^"]+)"', part["text"])
                                if versions and len(versions) > 0:
                                    # Ensure we have exactly 5 versions
                                    if len(versions) < 5:
                                        versions.extend([versions[-1]] * (5 - len(versions)))
                                    return versions[:5]
            
            # If we couldn't extract the refined versions using the expected structure
            logger.warning("Could not parse Gemini API response, returning original text as the only version")
            return ["Could not generate refined versions. Please check the API response."]
        
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return ["Error parsing response. Please try again."]

class MockTextRefiner:
    """
    Mock implementation of the text refiner for testing without API access.
    """
    
    def __init__(self):
        """Initialize the mock text refiner."""
        logger.info("Initialized mock text refiner")
        self.chunker = TextChunker()
    
    def chunk_text(self, raw_text: str) -> Dict[str, str]:
        """
        Split the raw text into logical chunks using the TextChunker.
        
        Args:
            raw_text (str): Raw extracted text from the prescription
            
        Returns:
            Dict[str, str]: Dictionary of chunks
        """
        return self.chunker.chunk_text(raw_text)
    
    def refine_chunk(self, chunk_type: str, chunk_text: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Mock refinement of a text chunk.
        
        Args:
            chunk_type (str): Type of the chunk
            chunk_text (str): Text content of the chunk
            context (Dict[str, Any], optional): Additional context for refinement
            
        Returns:
            List[str]: List of 5 mock refined versions of the chunk
        """
        logger.info(f"Mock refining {chunk_type} chunk")
        
        # Create 5 slightly different versions of the chunk
        if chunk_type == 'medications':
            return [
                "1. Amoxicillin 500mg\n   1 tablet 3 times daily for 10 days\n\n2. Ibuprofen 400mg\n   1 tablet every 6 hours as needed for pain",
                "1. Amoxicillin 500mg\n   One tablet three times daily for 10 days\n\n2. Ibuprofen 400mg\n   One tablet every 6 hours as needed for pain",
                "1. Amoxicillin 500mg\n   1 tab TID for 10 days\n\n2. Ibuprofen 400mg\n   1 tab q6h prn pain",
                "1. Amoxicillin 500mg\n   1 tablet TID × 10 days\n\n2. Ibuprofen 400mg\n   1 tablet q6h PRN pain",
                "1. Amoxicillin 500mg\n   One (1) tablet three times daily for ten days\n\n2. Ibuprofen 400mg\n   One (1) tablet every six hours as needed for pain"
            ]
        elif chunk_type == 'doctor_info':
            return [
                "Dr. John Smith\nMedical Center\n123 Health St.",
                "Dr. John Smith, MD\nMedical Center\n123 Health Street",
                "John Smith, M.D.\nMedical Center\n123 Health St.",
                "Dr. J. Smith\nMedical Center\n123 Health Street",
                "Doctor John Smith\nMedical Center\n123 Health Street"
            ]
        else:
            # For other chunk types, just return 5 copies of the original
            return [chunk_text] * 5

def get_text_refiner(use_mock: bool = False, api_key: Optional[str] = None) -> Tuple[TextChunker, GeminiTextRefiner]:
    """
    Factory function to get the appropriate text chunker and refiner.
    
    Args:
        use_mock (bool): Whether to use the mock refiner
        api_key (str, optional): Gemini API key
        
    Returns:
        Tuple[TextChunker, TextRefiner]: Instances of text chunker and refiner
    """
    chunker = TextChunker()
    
    if use_mock:
        mock_refiner = MockTextRefiner()
        return chunker, mock_refiner
    else:
        return chunker, GeminiTextRefiner(api_key)