#!/bin/bash

# Setup API Keys for Financial AI Platform

echo "🔑 Setting up API Keys..."
echo ""

# Alpha Vantage API Key
export ALPHA_VANTAGE_API_KEY=LODPOAHFH4DIACRR
echo "✓ Alpha Vantage API key set"

# Add to shell profile for persistence
SHELL_PROFILE=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
fi

if [ -n "$SHELL_PROFILE" ]; then
    if ! grep -q "ALPHA_VANTAGE_API_KEY" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Alpha Vantage API Key" >> "$SHELL_PROFILE"
        echo "export ALPHA_VANTAGE_API_KEY=LODPOAHFH4DIACRR" >> "$SHELL_PROFILE"
        echo "✓ Added to $SHELL_PROFILE for persistence"
    else
        echo "⚠️  API key already exists in $SHELL_PROFILE"
    fi
fi

echo ""
echo "✅ API key setup complete!"
echo ""
echo "To use immediately in this session, run:"
echo "  source setup_api_keys.sh"
echo ""
echo "Or restart your terminal to load from profile."

