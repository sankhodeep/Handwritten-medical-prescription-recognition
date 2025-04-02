import logging
import json
import os
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)

class OutputGenerator:
    """
    Class for generating the final structured output from the validated transcription.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the output generator.
        
        Args:
            output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized output generator with output directory: {output_dir}")
    
    def generate_output(self, transcription: Dict[str, Any], confidence_score: float, image_path: str) -> Dict[str, Any]:
        """
        Generate the final structured output.
        
        Args:
            transcription (Dict[str, Any]): Validated transcription dictionary
            confidence_score (float): Overall confidence score
            image_path (str): Path to the original prescription image
            
        Returns:
            Dict[str, Any]: Final output dictionary
        """
        try:
            # Create the output dictionary
            output = {
                "prescription_id": self._generate_prescription_id(image_path),
                "confidence_score": confidence_score,
                "transcription": self._format_transcription(transcription),
                "warnings": self._extract_warnings(transcription),
                "original_image": image_path
            }
            
            # Save the output to files
            self._save_output(output)
            
            logger.info("Successfully generated output")
            return output
        
        except Exception as e:
            logger.error(f"Error generating output: {str(e)}")
            raise
    
    def _generate_prescription_id(self, image_path: str) -> str:
        """
        Generate a unique ID for the prescription based on the image path.
        
        Args:
            image_path (str): Path to the prescription image
            
        Returns:
            str: Unique prescription ID
        """
        # Extract the filename without extension
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        
        # Add a timestamp to ensure uniqueness
        import time
        timestamp = int(time.time())
        
        return f"{name}_{timestamp}"
    
    def _format_transcription(self, transcription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the transcription for the final output.
        
        Args:
            transcription (Dict[str, Any]): Validated transcription dictionary
            
        Returns:
            Dict[str, Any]: Formatted transcription
        """
        formatted = {}
        
        # Use corrected chunks if available, otherwise use the best refined version
        corrected_chunks = transcription.get("corrected_chunks", {})
        refined_chunks = transcription.get("refined_chunks", {})
        
        for chunk_type in set(list(corrected_chunks.keys()) + list(refined_chunks.keys())):
            # Get the text for this chunk
            chunk_text = ""
            
            if chunk_type in corrected_chunks:
                # Use the corrected version
                chunk_text = corrected_chunks[chunk_type]
            elif chunk_type in refined_chunks and refined_chunks[chunk_type]:
                # Use the first refined version
                chunk_text = refined_chunks[chunk_type][0]
            
            # Mark uncertain words if confidence is low
            chunk_text = self._mark_uncertain_words(chunk_text, transcription, chunk_type)
            
            # Add to formatted transcription
            formatted[chunk_type] = chunk_text
        
        return formatted
    
    def _mark_uncertain_words(self, text: str, transcription: Dict[str, Any], chunk_type: str) -> str:
        """
        Mark uncertain words in the text with red color.
        
        Args:
            text (str): Text to mark
            transcription (Dict[str, Any]): Validated transcription dictionary
            chunk_type (str): Type of the chunk
            
        Returns:
            str: Text with uncertain words marked
        """
        # Get the confidence for this chunk
        chunk_confidence = 1.0  # Default to high confidence
        
        if "validation" in transcription:
            validation = transcription["validation"]
            if "chunk_confidences" in validation and chunk_type in validation["chunk_confidences"]:
                chunk_confidence = validation["chunk_confidences"][chunk_type]
        
        # If confidence is high, return the text as is
        if chunk_confidence > 0.8:
            return text
        
        # Get the refined versions for this chunk
        refined_versions = transcription.get("refined_chunks", {}).get(chunk_type, [])
        
        # If we don't have multiple refined versions, return the text as is
        if len(refined_versions) < 2:
            return text
        
        # Find words that differ between versions
        uncertain_words = self._find_uncertain_words(refined_versions)
        
        # Mark uncertain words in the text
        marked_text = text
        for word in uncertain_words:
            # Replace the word with a marked version
            # In a real implementation, this would use HTML or another format for red text
            # Here we'll use a simple format: [RED]word[/RED]
            marked_text = re.sub(
                r'\b' + re.escape(word) + r'\b',
                f"[RED]{word}[/RED]",
                marked_text
            )
        
        # Replace completely unreadable sections with "..."
        if chunk_confidence < 0.5:
            # Find sections that are completely different between versions
            unreadable_sections = self._find_unreadable_sections(refined_versions)
            
            for section in unreadable_sections:
                # Replace the section with "..."
                marked_text = marked_text.replace(section, "...")
        
        return marked_text
    
    def _find_uncertain_words(self, versions: List[str]) -> List[str]:
        """
        Find words that differ between versions.
        
        Args:
            versions (List[str]): List of text versions
            
        Returns:
            List[str]: List of uncertain words
        """
        # This is a simplified implementation
        # In a real system, you would use more sophisticated NLP techniques
        
        # Split each version into words
        version_words = [re.findall(r'\b\w+\b', version.lower()) for version in versions]
        
        # Find words that appear in some versions but not others
        all_words = set()
        for words in version_words:
            all_words.update(words)
        
        uncertain_words = []
        for word in all_words:
            # Count how many versions contain this word
            count = sum(1 for words in version_words if word in words)
            
            # If the word appears in some but not all versions, it's uncertain
            if 0 < count < len(versions):
                uncertain_words.append(word)
        
        return uncertain_words
    
    def _find_unreadable_sections(self, versions: List[str]) -> List[str]:
        """
        Find sections that are completely different between versions.
        
        Args:
            versions (List[str]): List of text versions
            
        Returns:
            List[str]: List of unreadable sections
        """
        # This is a simplified implementation
        # In a real system, you would use more sophisticated NLP techniques
        
        # For now, we'll just return an empty list
        return []
    
    def _extract_warnings(self, transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract warnings from the transcription.
        
        Args:
            transcription (Dict[str, Any]): Validated transcription dictionary
            
        Returns:
            List[Dict[str, Any]]: List of warnings
        """
        warnings = []
        
        # Extract flags from validation
        if "validation" in transcription and "flags" in transcription["validation"]:
            warnings = transcription["validation"]["flags"]
        
        return warnings
    
    def _save_output(self, output: Dict[str, Any]) -> None:
        """
        Save the output to files.
        
        Args:
            output (Dict[str, Any]): Output dictionary
        """
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save the output as JSON
            prescription_id = output["prescription_id"]
            json_path = os.path.join(self.output_dir, f"{prescription_id}.json")
            
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            # Save the formatted text output
            text_path = os.path.join(self.output_dir, f"{prescription_id}.txt")
            
            with open(text_path, 'w') as f:
                f.write(self._generate_text_output(output))
            
            logger.info(f"Saved output to {json_path} and {text_path}")
        
        except Exception as e:
            logger.error(f"Error saving output: {str(e)}")
            raise
    
    def _generate_text_output(self, output: Dict[str, Any]) -> str:
        """
        Generate a formatted text output.
        
        Args:
            output (Dict[str, Any]): Output dictionary
            
        Returns:
            str: Formatted text output
        """
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
        
        # Add note about uncertain words
        text += "\nNote: Words marked with [RED]...[/RED] have low confidence. "
        text += "Sections marked with ... are completely unreadable.\n"
        
        return text

class HTMLOutputGenerator(OutputGenerator):
    """
    Class for generating HTML output from the validated transcription.
    """
    
    def _save_output(self, output: Dict[str, Any]) -> None:
        """
        Save the output to files, including an HTML file.
        
        Args:
            output (Dict[str, Any]): Output dictionary
        """
        # Call the parent method to save JSON and text files
        super()._save_output(output)
        
        try:
            # Save the HTML output
            prescription_id = output["prescription_id"]
            html_path = os.path.join(self.output_dir, f"{prescription_id}.html")
            
            with open(html_path, 'w') as f:
                f.write(self._generate_html_output(output))
            
            logger.info(f"Saved HTML output to {html_path}")
        
        except Exception as e:
            logger.error(f"Error saving HTML output: {str(e)}")
            raise
    
    def _generate_html_output(self, output: Dict[str, Any]) -> str:
        """
        Generate an HTML output.
        
        Args:
            output (Dict[str, Any]): Output dictionary
            
        Returns:
            str: HTML output
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Prescription Transcription: {output['prescription_id']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            max-width: 800px;
        }}
        h1 {{
            color: #2c3e50;
        }}
        .section {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .section-title {{
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }}
        .uncertain {{
            color: red;
            font-weight: bold;
        }}
        .unreadable {{
            font-style: italic;
            color: #7f8c8d;
        }}
        .warning {{
            background-color: #fcf8e3;
            border-left: 5px solid #f39c12;
            padding: 10px;
            margin-bottom: 10px;
        }}
        .error {{
            background-color: #f2dede;
            border-left: 5px solid #e74c3c;
            padding: 10px;
            margin-bottom: 10px;
        }}
        .danger {{
            background-color: #f2dede;
            border-left: 5px solid #c0392b;
            padding: 10px;
            margin-bottom: 10px;
        }}
        .confidence {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .image-container {{
            margin-top: 20px;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Prescription Transcription</h1>
    <div class="confidence">
        <strong>Prescription ID:</strong> {output['prescription_id']}<br>
        <strong>Confidence Score:</strong> {output['confidence_score']:.2f}
    </div>
"""
        
        # Add each section of the transcription
        for section, content in output["transcription"].items():
            # Replace [RED]...[/RED] with HTML span
            content = content.replace("[RED]", '<span class="uncertain">').replace("[/RED]", '</span>')
            # Replace ... with HTML span
            content = content.replace("...", '<span class="unreadable">...</span>')
            
            html += f"""
    <div class="section">
        <div class="section-title">{section.upper()}</div>
        <div class="content">{content.replace("\n", "<br>")}</div>
    </div>"""
        
        # Add warnings if any
        if output["warnings"]:
            html += """
    <h2>Warnings</h2>"""
            
            for warning in output["warnings"]:
                warning_type = warning['type'].lower()
                html += f"""
    <div class="{warning_type}">
        <strong>{warning_type.upper()}:</strong> {warning['description']} (in {warning['location']})
    </div>"""
        
        # Add the original image
        html += f"""
    <div class="image-container">
        <h2>Original Prescription Image</h2>
        <img src="{os.path.relpath(output['original_image'], self.output_dir)}" alt="Original Prescription">
    </div>
    
    <div class="note">
        <p>Note: Words in <span class="uncertain">red</span> have low confidence. 
        Sections marked with <span class="unreadable">...</span> are completely unreadable.</p>
    </div>
</body>
</html>"""
        
        return html

def get_output_generator(output_format: str = "text", output_dir: str = "output") -> OutputGenerator:
    """
    Factory function to get the appropriate output generator.
    
    Args:
        output_format (str): Output format ("text", "html", or "both")
        output_dir (str): Directory to save output files
        
    Returns:
        OutputGenerator: An instance of an output generator
    """
    if output_format.lower() == "html":
        return HTMLOutputGenerator(output_dir)
    else:
        return OutputGenerator(output_dir)