import os
import logging
import json
import requests
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class MCPContextRetriever:
    """
    Class for retrieving contextual information using the Model Context Protocol (MCP).
    This implementation provides a flexible interface for connecting to various medical databases.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MCP context retriever.
        
        Args:
            config_path (str, optional): Path to the MCP configuration file
        """
        self.config = self._load_config(config_path)
        self.data_sources = self._initialize_data_sources()
        logger.info("Initialized MCP context retriever")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the MCP configuration from a file or use default configuration.
        
        Args:
            config_path (str, optional): Path to the configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            "data_sources": {
                "medications": {
                    "type": "local",
                    "path": "data/medications.json"
                },
                "doctors": {
                    "type": "local",
                    "path": "data/doctors.json"
                },
                "symptoms": {
                    "type": "local",
                    "path": "data/symptoms.json"
                }
            },
            "api": {
                "base_url": "https://api.example.com/medical",
                "api_key": os.environ.get("MEDICAL_API_KEY", "")
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded MCP configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {str(e)}")
                logger.warning("Using default configuration")
                return default_config
        else:
            logger.info("Using default MCP configuration")
            return default_config
    
    def _initialize_data_sources(self) -> Dict[str, Any]:
        """
        Initialize the data sources based on the configuration.
        
        Returns:
            Dict[str, Any]: Dictionary of initialized data sources
        """
        data_sources = {}
        
        for source_name, source_config in self.config["data_sources"].items():
            source_type = source_config.get("type", "local")
            
            if source_type == "local":
                # Initialize local data source
                path = source_config.get("path", "")
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            data_sources[source_name] = json.load(f)
                        logger.info(f"Loaded local data source '{source_name}' from {path}")
                    except Exception as e:
                        logger.error(f"Error loading local data source '{source_name}' from {path}: {str(e)}")
                        data_sources[source_name] = self._get_default_data(source_name)
                else:
                    logger.warning(f"Local data source path '{path}' does not exist")
                    data_sources[source_name] = self._get_default_data(source_name)
            
            elif source_type == "api":
                # For API data sources, we'll initialize them when they're first accessed
                data_sources[source_name] = {"type": "api", "config": source_config}
                logger.info(f"Initialized API data source '{source_name}'")
            
            else:
                logger.warning(f"Unknown data source type '{source_type}' for '{source_name}'")
                data_sources[source_name] = self._get_default_data(source_name)
        
        return data_sources
    
    def _get_default_data(self, source_name: str) -> Dict[str, Any]:
        """
        Get default data for a data source.
        
        Args:
            source_name (str): Name of the data source
            
        Returns:
            Dict[str, Any]: Default data
        """
        if source_name == "medications":
            return {
                "common_medications": [
                    {"name": "Amoxicillin", "dosages": ["250mg", "500mg"], "forms": ["tablet", "capsule", "suspension"]},
                    {"name": "Ibuprofen", "dosages": ["200mg", "400mg", "600mg", "800mg"], "forms": ["tablet", "capsule"]},
                    {"name": "Lisinopril", "dosages": ["5mg", "10mg", "20mg"], "forms": ["tablet"]},
                    {"name": "Metformin", "dosages": ["500mg", "850mg", "1000mg"], "forms": ["tablet"]},
                    {"name": "Atorvastatin", "dosages": ["10mg", "20mg", "40mg", "80mg"], "forms": ["tablet"]}
                ]
            }
        elif source_name == "doctors":
            return {
                "specialties": [
                    {"name": "Cardiology", "abbreviation": "Card"},
                    {"name": "Dermatology", "abbreviation": "Derm"},
                    {"name": "Endocrinology", "abbreviation": "Endo"},
                    {"name": "Gastroenterology", "abbreviation": "GI"},
                    {"name": "Neurology", "abbreviation": "Neuro"},
                    {"name": "Oncology", "abbreviation": "Onc"},
                    {"name": "Orthopedics", "abbreviation": "Ortho"},
                    {"name": "Pediatrics", "abbreviation": "Peds"},
                    {"name": "Psychiatry", "abbreviation": "Psych"},
                    {"name": "Pulmonology", "abbreviation": "Pulm"}
                ]
            }
        elif source_name == "symptoms":
            return {
                "common_symptoms": [
                    {"name": "Fever", "related_conditions": ["Infection", "Inflammation"]},
                    {"name": "Headache", "related_conditions": ["Migraine", "Tension", "Sinusitis"]},
                    {"name": "Cough", "related_conditions": ["Cold", "Flu", "Bronchitis", "Pneumonia"]},
                    {"name": "Fatigue", "related_conditions": ["Anemia", "Depression", "Hypothyroidism"]},
                    {"name": "Nausea", "related_conditions": ["Gastroenteritis", "Food poisoning", "Migraine"]}
                ]
            }
        else:
            return {}
    
    def retrieve_context(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve contextual information based on the query.
        
        Args:
            query (Dict[str, Any]): Query dictionary with keys like 'chunk_type', 'text', etc.
            
        Returns:
            Dict[str, Any]: Retrieved contextual information
        """
        try:
            chunk_type = query.get("chunk_type", "")
            text = query.get("text", "")
            
            context = {}
            
            # Retrieve context based on chunk type
            if chunk_type == "medications":
                context["medications"] = self._retrieve_medication_context(text)
            elif chunk_type == "doctor_info":
                context["doctors"] = self._retrieve_doctor_context(text)
            elif chunk_type == "diagnosis":
                context["symptoms"] = self._retrieve_symptom_context(text)
                context["conditions"] = self._retrieve_condition_context(text)
            
            # Add any additional context that might be helpful
            if "patient_info" in query:
                context["patient"] = self._retrieve_patient_context(query["patient_info"])
            
            logger.info(f"Retrieved context for {chunk_type}")
            return context
        
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return {}
    
    def _retrieve_medication_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Retrieve medication context.
        
        Args:
            text (str): Text to retrieve context for
            
        Returns:
            List[Dict[str, Any]]: List of medication context dictionaries
        """
        try:
            # Get the medications data source
            medications_data = self.data_sources.get("medications", {})
            
            # If it's an API data source, fetch the data
            if isinstance(medications_data, dict) and medications_data.get("type") == "api":
                medications_data = self._fetch_from_api("medications", text)
            
            # Extract medication names from the text
            # This is a simplified implementation; in a real system, you would use NLP
            words = text.lower().split()
            
            # Find matching medications
            matches = []
            for medication in medications_data.get("common_medications", []):
                med_name = medication["name"].lower()
                if med_name in text.lower() or any(med_name in word for word in words):
                    matches.append(medication)
            
            return matches
        
        except Exception as e:
            logger.error(f"Error retrieving medication context: {str(e)}")
            return []
    
    def _retrieve_doctor_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Retrieve doctor context.
        
        Args:
            text (str): Text to retrieve context for
            
        Returns:
            List[Dict[str, Any]]: List of doctor context dictionaries
        """
        try:
            # Get the doctors data source
            doctors_data = self.data_sources.get("doctors", {})
            
            # If it's an API data source, fetch the data
            if isinstance(doctors_data, dict) and doctors_data.get("type") == "api":
                doctors_data = self._fetch_from_api("doctors", text)
            
            # Extract specialty names from the text
            # This is a simplified implementation; in a real system, you would use NLP
            words = text.lower().split()
            
            # Find matching specialties
            matches = []
            for specialty in doctors_data.get("specialties", []):
                spec_name = specialty["name"].lower()
                spec_abbr = specialty["abbreviation"].lower()
                if spec_name in text.lower() or spec_abbr in text.lower() or any(spec_name in word for word in words):
                    matches.append(specialty)
            
            return matches
        
        except Exception as e:
            logger.error(f"Error retrieving doctor context: {str(e)}")
            return []
    
    def _retrieve_symptom_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Retrieve symptom context.
        
        Args:
            text (str): Text to retrieve context for
            
        Returns:
            List[Dict[str, Any]]: List of symptom context dictionaries
        """
        try:
            # Get the symptoms data source
            symptoms_data = self.data_sources.get("symptoms", {})
            
            # If it's an API data source, fetch the data
            if isinstance(symptoms_data, dict) and symptoms_data.get("type") == "api":
                symptoms_data = self._fetch_from_api("symptoms", text)
            
            # Extract symptom names from the text
            # This is a simplified implementation; in a real system, you would use NLP
            words = text.lower().split()
            
            # Find matching symptoms
            matches = []
            for symptom in symptoms_data.get("common_symptoms", []):
                symptom_name = symptom["name"].lower()
                if symptom_name in text.lower() or any(symptom_name in word for word in words):
                    matches.append(symptom)
            
            return matches
        
        except Exception as e:
            logger.error(f"Error retrieving symptom context: {str(e)}")
            return []
    
    def _retrieve_condition_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Retrieve condition context.
        
        Args:
            text (str): Text to retrieve context for
            
        Returns:
            List[Dict[str, Any]]: List of condition context dictionaries
        """
        # For now, we'll extract conditions from the symptoms data
        try:
            symptoms = self._retrieve_symptom_context(text)
            
            conditions = []
            for symptom in symptoms:
                for condition in symptom.get("related_conditions", []):
                    if condition.lower() in text.lower():
                        conditions.append({"name": condition})
            
            return conditions
        
        except Exception as e:
            logger.error(f"Error retrieving condition context: {str(e)}")
            return []
    
    def _retrieve_patient_context(self, patient_info: str) -> Dict[str, Any]:
        """
        Retrieve patient context.
        
        Args:
            patient_info (str): Patient information text
            
        Returns:
            Dict[str, Any]: Patient context dictionary
        """
        try:
            # Extract patient information
            # This is a simplified implementation; in a real system, you would use NLP
            patient = {}
            
            # Try to extract age
            import re
            age_match = re.search(r'age:?\s*(\d+)', patient_info, re.IGNORECASE)
            if age_match:
                patient["age"] = int(age_match.group(1))
            
            # Try to extract name
            name_match = re.search(r'(patient|name):?\s*([A-Za-z\s]+)', patient_info, re.IGNORECASE)
            if name_match:
                patient["name"] = name_match.group(2).strip()
            
            return patient
        
        except Exception as e:
            logger.error(f"Error retrieving patient context: {str(e)}")
            return {}
    
    def _fetch_from_api(self, source_name: str, query_text: str) -> Dict[str, Any]:
        """
        Fetch data from an API.
        
        Args:
            source_name (str): Name of the data source
            query_text (str): Text to query the API with
            
        Returns:
            Dict[str, Any]: API response data
        """
        try:
            source_config = self.data_sources.get(source_name, {}).get("config", {})
            api_url = source_config.get("url", self.config["api"]["base_url"] + f"/{source_name}")
            api_key = source_config.get("api_key", self.config["api"]["api_key"])
            
            # Prepare the request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "query": query_text
            }
            
            # Make the API request
            response = requests.post(api_url, headers=headers, json=payload)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return self._get_default_data(source_name)
        
        except Exception as e:
            logger.error(f"Error fetching from API: {str(e)}")
            return self._get_default_data(source_name)

class MockContextRetriever:
    """
    Mock implementation of the context retriever for testing without API access.
    """
    
    def __init__(self):
        """Initialize the mock context retriever."""
        logger.info("Initialized mock context retriever")
    
    def retrieve_context(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock retrieval of contextual information.
        
        Args:
            query (Dict[str, Any]): Query dictionary
            
        Returns:
            Dict[str, Any]: Mock contextual information
        """
        chunk_type = query.get("chunk_type", "")
        text = query.get("text", "")
        
        logger.info(f"Mock retrieving context for {chunk_type}")
        
        # Return mock context based on chunk type
        if chunk_type == "medications":
            return {
                "medications": [
                    {
                        "name": "Amoxicillin",
                        "dosages": ["250mg", "500mg"],
                        "forms": ["tablet", "capsule", "suspension"],
                        "common_usage": "Bacterial infections"
                    },
                    {
                        "name": "Ibuprofen",
                        "dosages": ["200mg", "400mg", "600mg", "800mg"],
                        "forms": ["tablet", "capsule"],
                        "common_usage": "Pain, inflammation, fever"
                    }
                ]
            }
        elif chunk_type == "doctor_info":
            return {
                "doctors": [
                    {
                        "specialty": "ENT",
                        "full_name": "Ear, Nose, and Throat",
                        "common_conditions": ["Sinusitis", "Tonsillitis", "Hearing loss"]
                    }
                ]
            }
        elif chunk_type == "diagnosis":
            return {
                "symptoms": [
                    {
                        "name": "Fever",
                        "related_conditions": ["Infection", "Inflammation"]
                    },
                    {
                        "name": "Headache",
                        "related_conditions": ["Migraine", "Tension", "Sinusitis"]
                    }
                ],
                "conditions": [
                    {
                        "name": "Sinusitis",
                        "common_medications": ["Amoxicillin", "Pseudoephedrine"]
                    }
                ]
            }
        else:
            return {}

def get_context_retriever(use_mock: bool = False, config_path: Optional[str] = None) -> Union[MCPContextRetriever, MockContextRetriever]:
    """
    Factory function to get the appropriate context retriever.
    
    Args:
        use_mock (bool): Whether to use the mock retriever
        config_path (str, optional): Path to the MCP configuration file
        
    Returns:
        ContextRetriever: An instance of a context retriever
    """
    if use_mock:
        return MockContextRetriever()
    else:
        return MCPContextRetriever(config_path)