# Handwritten Medical Prescription Recognition

An end-to-end system for automated medical prescription transcription using vision-language models (VLMs), OCR, LLMs, and structured retrieval systems.

## Overview

This project processes handwritten medical prescriptions and outputs structured, validated transcriptions. It follows a multi-step approach to ensure accuracy and contextual understanding of the prescriptions:

1. **Image Preprocessing**: Enhance prescription images to improve readability
2. **Handwriting Recognition**: Extract raw text from the processed image using Gemini Vision API
3. **Chunking & Refinement**: Split text into logical chunks and refine each chunk
4. **Contextual Retrieval**: Use Model Context Protocol (MCP) to retrieve relevant medical information
5. **Validation & Confidence Scoring**: Validate accuracy and generate confidence scores
6. **Final Output Generation**: Generate structured text output with confidence indicators

## Features

- **Enhanced Image Processing**: Increases contrast, removes noise, corrects skew, and segments handwritten text
- **Accurate Text Extraction**: Uses Gemini Vision API for initial text extraction
- **Intelligent Chunking**: Splits prescriptions into logical sections (doctor info, patient info, medications, etc.)
- **Multiple Refinement Versions**: Generates 5 different refined versions for each chunk to account for handwriting errors
- **Contextual Understanding**: Retrieves relevant medical information to improve recognition accuracy
- **Validation & Confidence Scoring**: Validates transcription accuracy and generates confidence scores
- **Structured Output**: Generates final output with uncertain words marked and completely unreadable sections indicated
- **Multiple Output Formats**: Supports text, JSON, and HTML output formats

## Installation

### Prerequisites

- Python 3.8 or higher
- Gemini API key (for production use)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/handwritten-medical-prescription-recognition.git
   cd handwritten-medical-prescription-recognition
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   ```
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Usage

Process a prescription image:

```
python src/main.py --image path/to/prescription.jpg
```

### Advanced Options

```
python src/main.py --image path/to/prescription.jpg --output-dir results --output-format html --verbose
```

Available options:
- `--image`: Path to the prescription image (required)
- `--output-dir`: Directory to save output files (default: output)
- `--output-format`: Output format: text, html, or both (default: both)
- `--use-mock`: Use mock implementations instead of actual APIs (for testing)
- `--api-key`: Gemini API key (can also be set via GEMINI_API_KEY env var)
- `--verbose`: Enable verbose logging

### Example

```
python src/main.py --image data/sample_images/prescription1.jpg --output-dir results --output-format both
```

## Project Structure

```
handwritten-medical-prescription-recognition/
├── data/
│   └── sample_images/       # Sample prescription images
├── src/
│   ├── preprocessing/       # Image preprocessing module
│   ├── recognition/         # Text extraction module
│   ├── refinement/          # Text chunking and refinement module
│   ├── retrieval/           # Contextual retrieval module
│   ├── validation/          # Validation and confidence scoring module
│   ├── output/              # Output generation module
│   └── main.py              # Main script
├── tests/                   # Unit tests
├── output/                  # Default output directory
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Modules

### 1. Image Preprocessing

Enhances prescription images to improve readability:
- Increases contrast
- Removes noise
- Corrects skew/rotation
- Segments handwritten text

### 2. Handwriting Recognition

Extracts raw text from the processed image using the Gemini Vision API.

### 3. Chunking & Refinement

Splits the raw text into logical chunks (doctor info, patient info, medications, etc.) and refines each chunk using a second-level LLM.

### 4. Contextual Retrieval

Uses the Model Context Protocol (MCP) to retrieve relevant medical information to improve recognition accuracy.

### 5. Validation & Confidence Scoring

Validates the transcription accuracy and generates confidence scores using a supervisor LLM.

### 6. Output Generation

Generates the final structured output with uncertain words marked and completely unreadable sections indicated.

## Extending the Project

### Adding New Data Sources

To add a new data source for contextual retrieval:

1. Create a JSON file with the data in the `data/` directory
2. Update the MCP configuration in `src/retrieval/context_retriever.py`

### Using Different Models

The project is designed to be model-agnostic. You can replace the Gemini API with other models by:

1. Creating a new class that implements the same interface
2. Updating the factory functions to return your new implementation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for vision and language capabilities
- OpenCV for image processing
- The medical community for domain expertise


## what the model says:
# Handwritten Medical Prescription Recognition System

I've successfully implemented a comprehensive system for automated medical prescription transcription using vision-language models, OCR, LLMs, and structured retrieval systems. The implementation follows the multi-step approach outlined in the original requirements:

## Implementation Overview

1. **Image Preprocessing Module** (`src/preprocessing/image_processor.py`)
   - Enhances prescription images by increasing contrast, removing noise, correcting skew, and segmenting handwritten text
   - Uses OpenCV and PIL for image processing

2. **Text Extraction Module** (`src/recognition/text_extractor.py`)
   - Extracts raw text from processed images using the Gemini Vision API
   - Includes a mock implementation for testing without API access

3. **Text Chunking & Refinement Module** (`src/refinement/text_refiner.py`)
   - Splits raw text into logical chunks (doctor info, patient info, medications, etc.)
   - Refines each chunk using a second-level LLM (Gemini API)
   - Generates multiple refined versions to account for handwriting errors

4. **Contextual Retrieval Module** (`src/retrieval/context_retriever.py`)
   - Implements the Model Context Protocol (MCP) for retrieving relevant medical information
   - Supports both local data sources and external APIs
   - Includes sample data for medications, doctors, and symptoms

5. **Validation Module** (`src/validation/prescription_validator.py`)
   - Validates transcription accuracy using a supervisor LLM
   - Generates confidence scores and flags potential errors
   - Ensures clinical reasonableness of prescriptions

6. **Output Generation Module** (`src/output/output_generator.py`)
   - Generates structured output with uncertain words marked
   - Supports multiple output formats (text, JSON, HTML)

7. **Main Orchestration Script** (`src/main.py`)
   - Coordinates the entire workflow
   - Provides a command-line interface for processing prescription images

## Additional Components

- **Demo Script** (`src/demo.py`): A simplified version that demonstrates the workflow without requiring OpenCV dependencies
- **Example Scripts**: Additional scripts to demonstrate specific functionality
- **Test Suite**: Basic tests to verify system functionality
- **Configuration Files**: For MCP and logging
- **Sample Data**: For contextual retrieval

## How to Use the System

### Basic Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the demo script to see the system in action:
   ```
   python src/demo.py
   ```

3. For production use with real prescription images, use the main script:
   ```
   python src/main.py --image path/to/prescription.jpg --api-key YOUR_GEMINI_API_KEY
   ```

### Advanced Usage

- Use the `--output-format` flag to specify the output format (text, html, or both)
- Use the `--output-dir` flag to specify the output directory
- Set the `GEMINI_API_KEY` environment variable instead of using the `--api-key` flag
- Use the `--verbose` flag for detailed logging

## Extending the System

The system is designed to be modular and extensible:

1. To use different models, create new classes that implement the same interfaces
2. To add new data sources for contextual retrieval, update the MCP configuration
3. To customize the output format, extend the OutputGenerator class

The implementation is well-documented, includes error handling, and follows best practices for clean, modular code.