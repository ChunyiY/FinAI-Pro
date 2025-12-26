# 🔄 Application Restart Guide

## Quick Restart

### Method 1: Using the startup script (Recommended)
```bash
./start_app.sh
```

This script will:
- ✅ Check Python version
- ✅ Verify all dependencies are installed
- ✅ Check code syntax
- ✅ Verify NLTK data
- ✅ Start the Streamlit application

### Method 2: Manual restart
```bash
# Stop any running Streamlit processes
pkill -f streamlit

# Verify code
python verify_code.py

# Check dependencies
python check_dependencies.py

# Start the application
streamlit run app.py
```

## Troubleshooting Restart Issues

### Issue: Port 8501 already in use

**Solution:**
```bash
# Find and kill the process using port 8501
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run app.py --server.port 8502
```

### Issue: Dependencies missing

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python check_dependencies.py
```

### Issue: NLTK data missing

**Solution:**
```bash
# Download NLTK data
python setup.py
```

### Issue: Code syntax errors

**Solution:**
```bash
# Verify code syntax
python verify_code.py

# Fix any reported errors
```

## Pre-Demo Checklist

Before starting your demo/interview:

- [ ] All dependencies installed (`python check_dependencies.py`)
- [ ] Code syntax verified (`python verify_code.py`)
- [ ] NLTK data downloaded (`python setup.py`)
- [ ] Test data fetched (to avoid rate limits during demo)
- [ ] Application starts successfully
- [ ] All pages load correctly

## Common Commands

```bash
# Check everything is ready
python check_dependencies.py && python verify_code.py

# Start application
streamlit run app.py

# Start with custom port
streamlit run app.py --server.port 8502

# Start with browser disabled (for headless)
streamlit run app.py --server.headless true
```

## Application URLs

Once started, the application will be available at:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501 (shown in terminal)

## Stopping the Application

- Press `Ctrl+C` in the terminal
- Or use: `pkill -f streamlit`

