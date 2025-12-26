# Quick Start Guide

**Step-by-step instructions to run FinAI Pro**

---

## Prerequisites

- Python 3.9 or higher
- Internet connection
- (Optional) Alpha Vantage API key for real-time data

---

## Installation

### Step 1: Install Dependencies

Open Terminal (Mac/Linux) or Command Prompt (Windows) and run:

```bash
pip install -r requirements.txt
```

**If you see errors:**
- Try: `pip3 install -r requirements.txt` (Mac/Linux)
- Or: `python -m pip install -r requirements.txt`

### Step 2: Configure API Key (Optional but Recommended)

1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Copy `config.py.example` to `config.py`:
   ```bash
   cp config.py.example config.py
   ```
3. Edit `config.py` and add your API key:
   ```python
   ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"
   ```

### Step 3: Download NLTK Data (First Time Only)

```bash
python setup.py
```

---

## Running the Application

### Method 1: Using the Startup Script (Recommended)

```bash
./scripts/start.sh
```

### Method 2: Direct Command

```bash
streamlit run app.py
```

The application will open at: **http://localhost:8501**

---

## First Steps

1. **Open the application** in your browser
2. **Navigate to "Cross-Sectional Alpha Engine"** (key feature)
3. **Enter stock symbols**: `AAPL,MSFT,GOOGL,AMZN,TSLA`
4. **Click "Generate Cross-Sectional Alpha"**
5. **Wait 2-3 minutes** for model training
6. **View results**: Alpha signals, risk adjustments, and portfolio weights

---

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "API rate limit exceeded"
- Wait 1-2 minutes and try again
- Or get your own free API key from Alpha Vantage

### Browser doesn't open
- Manually open: `http://localhost:8501`

### "Insufficient data" error
- Select longer data period (2y instead of 1y)
- Try different stock symbols

---

## Stop the Application

Press `Ctrl + C` in the Terminal/Command Prompt

---

## Next Steps

- See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed feature demonstrations
- See [README.md](README.md) for technical documentation
