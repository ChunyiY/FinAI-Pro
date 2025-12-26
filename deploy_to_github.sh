#!/bin/bash

# FinAI Pro - GitHub Deployment Script
# This script initializes git and prepares the repository for GitHub

set -e

echo "🚀 FinAI Pro - GitHub Deployment"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository
if [ ! -d ".git" ]; then
    echo -e "${BLUE}📦 Initializing git repository...${NC}"
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
else
    echo -e "${YELLOW}⚠️  Git repository already exists${NC}"
fi

# Add all files
echo -e "${BLUE}📝 Staging files...${NC}"
git add .

# Check if config.py exists and warn user
if [ -f "config.py" ]; then
    echo -e "${YELLOW}⚠️  Warning: config.py detected. Make sure it doesn't contain sensitive API keys!${NC}"
    echo -e "${YELLOW}   Consider using config.py.example instead.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Create initial commit
echo -e "${BLUE}💾 Creating initial commit...${NC}"
git commit -m "Initial commit: FinAI Pro - Enterprise Financial Intelligence Platform

✨ Features:
- Stock price prediction using LSTM deep learning
- Financial news sentiment analysis with NLP
- Portfolio optimization with Modern Portfolio Theory
- Real-time market data analysis with technical indicators
- Professional UI with enterprise-grade design system

🛠️ Tech Stack:
- Python 3.9+
- Streamlit
- PyTorch
- Alpha Vantage API
- Plotly visualizations" || {
    echo -e "${YELLOW}⚠️  No changes to commit or commit failed${NC}"
}

echo ""
echo -e "${GREEN}✅ Repository ready for GitHub!${NC}"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Add the remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/finai-pro.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Or use SSH:"
echo "   git remote add origin git@github.com:YOUR_USERNAME/finai-pro.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🎉 Your professional GitHub repo is ready!"

