#!/bin/bash

# FinAI Pro - Push to GitHub
# Run this script after creating your GitHub repository

set -e

echo "🚀 FinAI Pro - Push to GitHub"
echo "=============================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: FinAI Pro - Enterprise Financial Intelligence Platform

✨ Features:
- Stock price prediction using LSTM deep learning
- Financial news sentiment analysis with NLP
- Portfolio optimization with Modern Portfolio Theory
- Real-time market data analysis
- Professional UI with enterprise design system

🛠️ Tech Stack: Python, Streamlit, PyTorch, Alpha Vantage API"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

echo ""
echo "📋 Next steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo "   Name: finai-pro (or your choice)"
echo "   Description: Enterprise Financial Intelligence Platform"
echo "   Set to Public"
echo "   DO NOT initialize with README"
echo ""
echo "2. After creating the repo, run:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/finai-pro.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Or use SSH:"
echo "   git remote add origin git@github.com:YOUR_USERNAME/finai-pro.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🎉 Your professional GitHub repo is ready to push!"

