#!/bin/bash

# FinAI Pro - Git Push Script
# This script helps push changes to GitHub

echo "=========================================="
echo "  FinAI Pro - Git Push to GitHub"
echo "=========================================="
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository"
    echo "Initialize with: git init"
    exit 1
fi

# Check git status
echo "Checking git status..."
git status --short

echo ""
read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Add all changes
echo ""
echo "Adding all changes..."
git add .

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --short

echo ""
read -p "Commit these changes? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Commit
echo ""
echo "Committing changes..."
git commit -m "Add Cross-Sectional Alpha Engine and restructure project

- Add industry-grade cross-sectional alpha engine module (alpha/)
- Implement risk adjustment layer (risk/)
- Add portfolio allocation layer (portfolio/)
- Restructure project with standard directory layout (docs/, scripts/, tests/)
- Update README with institutional-grade documentation
- Add comprehensive demo guide (DEMO_GUIDE.md) for non-technical users
- Organize documentation and scripts into proper directories
- Add startup script for easy deployment"

# Check if remote exists
if ! git remote | grep -q "origin"; then
    echo ""
    echo "⚠️  No remote 'origin' found."
    echo "Add remote with: git remote add origin <your-repo-url>"
    exit 1
fi

# Push
echo ""
echo "Pushing to GitHub..."
BRANCH=$(git branch --show-current)
echo "Branch: $BRANCH"

git push origin $BRANCH

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
else
    echo ""
    echo "❌ Push failed. Try:"
    echo "   git pull origin $BRANCH --rebase"
    echo "   git push origin $BRANCH"
fi
