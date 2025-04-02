#!/bin/bash
# Demo script for the Handwritten Medical Prescription Recognition system
# This script demonstrates the complete workflow of the system

# Set up colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}  Handwritten Medical Prescription Recognition Demo${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Check if a sample image exists
SAMPLE_IMAGE="data/sample_images/sample_prescription.jpg"
if [ ! -f "$SAMPLE_IMAGE" ]; then
    echo -e "${YELLOW}Sample image not found at $SAMPLE_IMAGE${NC}"
    echo -e "${YELLOW}Please add a sample prescription image before running this demo.${NC}"
    echo -e "${YELLOW}See data/sample_images/README.txt for instructions.${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Step 1: Install dependencies
echo -e "\n${GREEN}Step 1: Installing dependencies...${NC}"
pip install -r requirements.txt

# Step 2: Run the example MCP retrieval script
echo -e "\n${GREEN}Step 2: Testing MCP context retrieval...${NC}"
python src/mcp_example.py

# Step 3: Run the main prescription processing script with mock implementation
echo -e "\n${GREEN}Step 3: Processing prescription with mock implementation...${NC}"
python src/main.py --image "$SAMPLE_IMAGE" --output-dir output --use-mock

# Step 4: Check the output files
echo -e "\n${GREEN}Step 4: Checking output files...${NC}"
echo -e "${BLUE}Output files:${NC}"
ls -la output/

# Step 5: Display the text output
echo -e "\n${GREEN}Step 5: Displaying text output...${NC}"
TEXT_OUTPUT=$(find output -name "*.txt" | head -n 1)
if [ -f "$TEXT_OUTPUT" ]; then
    echo -e "${BLUE}Contents of $TEXT_OUTPUT:${NC}"
    cat "$TEXT_OUTPUT"
else
    echo -e "${RED}No text output file found.${NC}"
fi

# Step 6: Open the HTML output in the default browser (if available)
echo -e "\n${GREEN}Step 6: Opening HTML output...${NC}"
HTML_OUTPUT=$(find output -name "*.html" | head -n 1)
if [ -f "$HTML_OUTPUT" ]; then
    echo -e "${BLUE}HTML output available at: $HTML_OUTPUT${NC}"
    
    # Try to open the HTML file in the default browser
    if command -v xdg-open &> /dev/null; then
        echo -e "${BLUE}Opening HTML output in browser...${NC}"
        xdg-open "$HTML_OUTPUT"
    elif command -v open &> /dev/null; then
        echo -e "${BLUE}Opening HTML output in browser...${NC}"
        open "$HTML_OUTPUT"
    else
        echo -e "${YELLOW}Could not open browser automatically. Please open $HTML_OUTPUT manually.${NC}"
    fi
else
    echo -e "${RED}No HTML output file found.${NC}"
fi

echo -e "\n${GREEN}Demo completed!${NC}"
echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}  For production use, remove the --use-mock flag and set${NC}"
echo -e "${BLUE}  your Gemini API key with --api-key or GEMINI_API_KEY env var${NC}"
echo -e "${BLUE}===========================================================${NC}"