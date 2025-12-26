#!/bin/bash

# Financial AI Analysis Platform - Startup Script
# This script ensures all dependencies are installed and starts the application

set -e  # Exit on error

echo "🚀 Financial AI Analysis Platform - Startup Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

# Check if required packages are installed
echo ""
echo "📦 Checking dependencies..."
MISSING_PACKAGES=()

check_package() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} $1 installed"
    else
        echo -e "   ${RED}✗${NC} $1 not installed"
        MISSING_PACKAGES+=("$1")
    fi
}

check_package "streamlit"
check_package "pandas"
check_package "numpy"
check_package "yfinance"
check_package "plotly"
check_package "torch"
check_package "sklearn"
check_package "nltk"
check_package "textblob"
check_package "vaderSentiment"
check_package "scipy"

# Install missing packages if any
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Missing packages detected. Installing...${NC}"
    pip install -r requirements.txt
    echo ""
fi

# Verify code syntax
echo ""
echo "🔍 Verifying code syntax..."
if python3 verify_code.py; then
    echo -e "${GREEN}✓ Code verification passed${NC}"
else
    echo -e "${RED}✗ Code verification failed${NC}"
    exit 1
fi

# Check NLTK data
echo ""
echo "📥 Checking NLTK data..."
if python3 -c "import nltk; nltk.data.find('tokenizers/punkt')" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} NLTK punkt data found"
else
    echo -e "   ${YELLOW}⚠${NC}  NLTK punkt data missing, downloading..."
    python3 setup.py
fi

# Start the application
echo ""
echo "=================================================="
echo -e "${GREEN}✅ All checks passed!${NC}"
echo ""
echo "🚀 Starting Streamlit application..."
echo "   The app will open in your browser at: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the application"
echo "=================================================="
echo ""

streamlit run app.py

