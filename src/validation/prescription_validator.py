import os
import logging
import json
import requests
import base64
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class SupervisorLLM:
    """
    Class for validating prescription transcriptions using a supervisor LLM.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the supervisor LLM.
        
        Args:
            api_key (str, optional): Gemini API key. If not provided, it will try to get it from
                                     the GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No Gemini API key provided. Using placeholder for development.")
            self.api_key = "PLACEHOLDER_API_KEY"
        
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
        logger.info("Initialized supervisor LLM")
    
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
    
    def validate(self, transcription: Dict[str, Any], original_image_path: str) -> Tuple[Dict[str, Any], float]:
        """
        Validate the transcription and generate a confidence score.
        
        Args:
            transcription (Dict[str, Any]): Transcription dictionary with keys like 'chunks', 'refined_chunks', etc.
            original_image_path (str): Path to the original prescription image
            
        Returns:
            Tuple[Dict[str, Any], float]: Tuple of validated transcription and confidence score
        """
        try:
            # Encode the image
            encoded_image = self._encode_image(original_image_path)
            
            # Prepare the transcription text
            transcription_text = self._format_transcription_for_validation(transcription)
            
            # Prepare the request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": self._create_validation_prompt(transcription_text)
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": encoded_image
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 2048,
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
                
                # Parse the validation response
                validated_transcription, confidence_score = self._parse_validation_response(response_data, transcription)
                logger.info(f"Successfully validated transcription with confidence score {confidence_score}")
                return validated_transcription, confidence_score
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                # For development/testing, return the original transcription with a moderate confidence score
                if self.api_key == "PLACEHOLDER_API_KEY":
                    logger.warning("Using placeholder validation since API key is not set")
                    return transcription, 0.75
                raise Exception(f"API request failed: {response.text}")
        
        except Exception as e:
            logger.error(f"Error validating transcription: {str(e)}")
            # Return the original transcription with a low confidence score
            return transcription, 0.5
    
    def _format_transcription_for_validation(self, transcription: Dict[str, Any]) -> str:
        """
        Format the transcription for validation.
        
        Args:
            transcription (Dict[str, Any]): Transcription dictionary
            
        Returns:
            str: Formatted transcription text
        """
        formatted_text = "PRESCRIPTION TRANSCRIPTION:\n\n"
        
        # Add each chunk to the formatted text
        for chunk_type, refined_versions in transcription.get("refined_chunks", {}).items():
            # Use the first refined version for validation
            if refined_versions and len(refined_versions) > 0:
                chunk_text = refined_versions[0]
                formatted_text += f"--- {chunk_type.upper()} ---\n{chunk_text}\n\n"
        
        return formatted_text
    
    def _create_validation_prompt(self, transcription_text: str) -> str:
        """
        Create a prompt for the validation LLM.
        
        Args:
            transcription_text (str): Formatted transcription text
            
        Returns:
            str: Validation prompt
        """
        return f"""
        You are a medical expert tasked with validating the accuracy of a prescription transcription.
        
        I will provide you with:
        1. A transcription of a handwritten medical prescription
        2. The original prescription image
        
        Your task is to:
        1. Validate the accuracy of the transcription by comparing it with the image
        2. Identify any errors or inconsistencies
        3. Provide corrections for any errors
        4. Generate a confidence score (0-100%) for the overall transcription
        5. Flag any suspicious or potentially dangerous errors
        
        Here is the transcription:
        
        {transcription_text}
        
        Please analyze the transcription and the image, then provide your validation in the following JSON format:
        
        ```json
        {{
            "validation": {{
                "doctor_info": {{
                    "is_accurate": true/false,
                    "corrections": "corrected text if needed",
                    "confidence": 0-100
                }},
                "patient_info": {{
                    "is_accurate": true/false,
                    "corrections": "corrected text if needed",
                    "confidence": 0-100
                }},
                "medications": {{
                    "is_accurate": true/false,
                    "corrections": "corrected text if needed",
                    "confidence": 0-100
                }},
                "diagnosis": {{
                    "is_accurate": true/false,
                    "corrections": "corrected text if needed",
                    "confidence": 0-100
                }},
                "instructions": {{
                    "is_accurate": true/false,
                    "corrections": "corrected text if needed",
                    "confidence": 0-100
                }},
                "referral": {{
                    "is_accurate": true/false,
                    "corrections": "corrected text if needed",
                    "confidence": 0-100
                }},
                "signature": {{
                    "is_accurate": true/false,
                    "corrections": "corrected text if needed",
                    "confidence": 0-100
                }}
            }},
            "overall_confidence": 0-100,
            "flags": [
                {{
                    "type": "error/warning/danger",
                    "description": "Description of the issue",
                    "location": "Which part of the prescription has the issue"
                }}
            ]
        }}
        ```
        
        Only include sections in the validation that are present in the transcription.
        For the overall confidence, consider the clinical reasonableness of the prescription based on:
        - Are the medicine names, dosages, and schedules clinically reasonable?
        - Are there any inconsistencies in the extracted text?
        - Is the prescription appropriate for the patient's age, symptoms, and the doctor's specialty?
        
        Provide only the JSON response, no additional text.
        """
    
    def _parse_validation_response(self, response_data: Dict[str, Any], original_transcription: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Parse the validation response from the LLM.
        
        Args:
            response_data (Dict[str, Any]): Response data from the LLM
            original_transcription (Dict[str, Any]): Original transcription dictionary
            
        Returns:
            Tuple[Dict[str, Any], float]: Tuple of validated transcription and confidence score
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
                                # Extract JSON from the text (it might be wrapped in ```json ... ```)
                                text = part["text"].strip()
                                json_match = text
                                if "```json" in text:
                                    json_match = text.split("```json")[1].split("```")[0].strip()
                                elif "```" in text:
                                    json_match = text.split("```")[1].split("```")[0].strip()
                                
                                validation_data = json.loads(json_match)
                                
                                # Extract the validation information
                                validation = validation_data.get("validation", {})
                                overall_confidence = validation_data.get("overall_confidence", 0) / 100.0  # Convert to 0-1 scale
                                flags = validation_data.get("flags", [])
                                
                                # Update the transcription with corrections
                                validated_transcription = self._apply_corrections(original_transcription, validation)
                                
                                # Add validation information to the transcription
                                validated_transcription["validation"] = {
                                    "confidence": overall_confidence,
                                    "flags": flags
                                }
                                
                                return validated_transcription, overall_confidence
                            
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing validation response as JSON: {str(e)}")
                                # Return the original transcription with a moderate confidence score
                                return original_transcription, 0.6
            
            # If we couldn't extract the validation information
            logger.warning("Could not parse validation response, returning original transcription")
            return original_transcription, 0.6
        
        except Exception as e:
            logger.error(f"Error parsing validation response: {str(e)}")
            return original_transcription, 0.5
    
    def _apply_corrections(self, transcription: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply corrections to the transcription based on validation.
        
        Args:
            transcription (Dict[str, Any]): Original transcription dictionary
            validation (Dict[str, Any]): Validation dictionary
            
        Returns:
            Dict[str, Any]: Corrected transcription
        """
        corrected_transcription = transcription.copy()
        
        # Initialize corrected chunks if not present
        if "corrected_chunks" not in corrected_transcription:
            corrected_transcription["corrected_chunks"] = {}
        
        # Apply corrections to each chunk
        for chunk_type, chunk_validation in validation.items():
            if not chunk_validation.get("is_accurate", True) and "corrections" in chunk_validation:
                # Get the corrections
                corrections = chunk_validation["corrections"]
                
                # Update the corrected chunks
                corrected_transcription["corrected_chunks"][chunk_type] = corrections
        
        return corrected_transcription

class MockSupervisorLLM:
    """
    Mock implementation of the supervisor LLM for testing without API access.
    """
    
    def __init__(self):
        """Initialize the mock supervisor LLM."""
        logger.info("Initialized mock supervisor LLM")
    
    def validate(self, transcription: Dict[str, Any], original_image_path: str) -> Tuple[Dict[str, Any], float]:
        """
        Mock validation of the transcription.
        
        Args:
            transcription (Dict[str, Any]): Transcription dictionary
            original_image_path (str): Path to the original prescription image
            
        Returns:
            Tuple[Dict[str, Any], float]: Tuple of validated transcription and confidence score
        """
        logger.info("Mock validating transcription")
        
        # Create a copy of the transcription to modify
        validated_transcription = transcription.copy()
        
        # Add mock validation information
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
        
        # Add mock corrected chunks
        validated_transcription["corrected_chunks"] = {}
        for chunk_type, refined_versions in transcription.get("refined_chunks", {}).items():
            if refined_versions and len(refined_versions) > 0:
                # Use the first refined version as the corrected version
                validated_transcription["corrected_chunks"][chunk_type] = refined_versions[0]
        
        return validated_transcription, 0.85

def get_supervisor_llm(use_mock: bool = False, api_key: Optional[str] = None) -> SupervisorLLM:
    """
    Factory function to get the appropriate supervisor LLM.
    
    Args:
        use_mock (bool): Whether to use the mock supervisor
        api_key (str, optional): Gemini API key
        
    Returns:
        SupervisorLLM: An instance of a supervisor LLM
    """
    if use_mock:
        return MockSupervisorLLM()
    else:
        return SupervisorLLM(api_key)