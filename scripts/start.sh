#!/bin/bash

# FinAI Pro - Startup Script
# This script helps start the application easily

echo "=========================================="
echo "  FinAI Pro - Starting Application"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    echo "Please install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if requirements are installed
echo "Checking dependencies..."
if ! python -c "import streamlit" &> /dev/null; then
    echo "⚠️  Dependencies not installed. Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        exit 1
    fi
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies OK"
fi

# Check if config.py exists
if [ ! -f "config.py" ]; then
    echo "⚠️  config.py not found. Creating from template..."
    if [ -f "config.py.example" ]; then
        cp config.py.example config.py
        echo "✓ Created config.py (please add your API key if needed)"
    else
        echo "⚠️  config.py.example not found. Continuing without API key..."
    fi
fi

# Start the application
echo ""
echo "=========================================="
echo "  Starting Streamlit application..."
echo "=========================================="
echo ""
echo "The application will open in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app.py

