#!/bin/bash

# Zeus Splunk Query LLM - Quick Start Script
# This script helps you get started with Zeus quickly

set -e

echo "=========================================="
echo "Zeus Splunk Query LLM - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "ERROR: Python 3.9 or higher is required"
    echo "Current version: $(python3 --version)"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "(This may take a few minutes...)"
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Prepare sample data
echo "Preparing sample training data..."
python -m src.data_preparation.prepare_data \
    --input-dir data/raw \
    --output-dir data/processed \
    --format-type alpaca \
    --validate-only

echo "✓ Data validation complete"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Process training data:"
echo "   python -m src.data_preparation.prepare_data \\"
echo "     --input-dir data/raw \\"
echo "     --output-dir data/processed \\"
echo "     --format-type alpaca"
echo ""
echo "2. Fine-tune the model:"
echo "   python -m src.training.train \\"
echo "     --config configs/training_config.yaml"
echo ""
echo "3. Test the model:"
echo "   python examples/basic_usage.py"
echo ""
echo "4. Start the API server:"
echo "   python -m src.inference.server \\"
echo "     --model-path models/splunk-query-llm \\"
echo "     --port 8000"
echo ""
echo "For more information, see README.md"
echo ""
