# FinAI Pro | Demonstration Guide

**For Non-Technical Users**

This guide provides step-by-step instructions to run and demonstrate the FinAI Pro platform. No programming knowledge required.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Running the Application](#running-the-application)
4. [Feature Demonstrations](#feature-demonstrations)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- ✅ A computer with internet connection
- ✅ Python 3.9 or higher installed
  - Check: Open Terminal (Mac) or Command Prompt (Windows) and type `python --version`
  - If not installed: Download from [python.org](https://www.python.org/downloads/)
- ✅ Basic familiarity with using a web browser

**Optional but Recommended:**
- Alpha Vantage API key (free) for real-time data
  - Get one at: [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)

---

## Installation Steps

### Step 1: Download the Project

1. **If using Git:**
   ```bash
   git clone https://github.com/yourusername/finai-pro.git
   cd finai-pro
   ```

2. **If downloading as ZIP:**
   - Click "Download ZIP" on GitHub
   - Extract the ZIP file to a folder (e.g., `Desktop/finai-pro`)

### Step 2: Open Terminal/Command Prompt

- **Mac/Linux**: Open "Terminal" application
- **Windows**: Open "Command Prompt" or "PowerShell"

Navigate to the project folder:
```bash
cd /path/to/finai-pro
# Example: cd ~/Desktop/finai-pro
```

### Step 3: Install Dependencies

Type the following command and press Enter:

```bash
pip install -r requirements.txt
```

**What this does:** Installs all required Python packages (takes 2-5 minutes)

**If you see errors:**
- Try: `pip3 install -r requirements.txt` (Mac/Linux)
- Or: `python -m pip install -r requirements.txt`

### Step 4: Configure API Key (Optional)

**For real-time data (recommended):**

1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Sign up (takes 1 minute)
   - Copy your API key

2. Create `config.py` file:
   ```bash
   cp config.py.example config.py
   ```

3. Open `config.py` in a text editor and replace:
   ```python
   ALPHA_VANTAGE_API_KEY = "your_api_key_here"
   ```
   with your actual API key:
   ```python
   ALPHA_VANTAGE_API_KEY = "YOUR_ACTUAL_KEY_HERE"
   ```

**Note:** The platform works without an API key but with limited data.

### Step 5: Download NLTK Data (First Time Only)

Type:
```bash
python setup.py
```

**What this does:** Downloads language data for sentiment analysis (one-time setup)

---

## Running the Application

### Start the Platform

Type the following command:

```bash
streamlit run app.py
```

**What happens:**
1. The application starts
2. Your web browser should automatically open
3. If not, open: `http://localhost:8501`

**You should see:**
- FinAI Pro welcome page
- Navigation sidebar on the left
- Multiple feature modules

### Stop the Application

- Press `Ctrl + C` in the Terminal/Command Prompt
- Or close the browser tab

---

## Feature Demonstrations

### Demo 1: Stock Price Prediction (LSTM)

**Purpose:** Predict future stock prices using deep learning

**Steps:**
1. Click **"LSTM Price Forecast"** in the sidebar
2. Enter a stock symbol (e.g., `AAPL` for Apple)
3. Select data period: `2y` (recommended)
4. Set prediction days: `30`
5. Click **"Generate Prediction"**

**What to show:**
- Model training progress
- Performance metrics (RMSE, MAE, MAPE)
- Interactive chart showing historical and predicted prices
- Prediction statistics (high, low, change)

**Expected time:** 1-2 minutes

---

### Demo 2: Cross-Sectional Alpha Engine ⭐ **Key Feature**

**Purpose:** Generate risk-adjusted portfolio weights using relative value ranking

**Steps:**
1. Click **"Cross-Sectional Alpha Engine"** in the sidebar
2. Enter multiple stock symbols (comma-separated):
   ```
   AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA
   ```
3. Select data period: `1y`
4. Keep default settings (or adjust if needed)
5. Click **"Generate Cross-Sectional Alpha"**

**What to show:**
- Model training with Rank IC metric
- Results table showing:
  - **Alpha Raw**: Raw ML predictions
  - **Alpha Z-Score**: Normalized scores
  - **Volatility**: Stock volatility
  - **Risk-Adjusted Alpha**: Final signal after risk adjustment
  - **Weight**: Portfolio allocation weight
- Top 5 and Bottom 5 stocks
- Alpha distribution chart
- Portfolio weight pie chart

**Key Points to Explain:**
- "This ranks stocks relative to each other on the same day"
- "Risk adjustment is mandatory - raw alpha is too noisy"
- "Final output is portfolio weights, not price predictions"
- "This is how institutional quant funds allocate risk"

**Expected time:** 2-3 minutes

---

### Demo 3: Sentiment Analysis

**Purpose:** Analyze financial news sentiment

**Steps:**
1. Click **"Sentiment Analysis"** in the sidebar
2. Enter stock symbol: `AAPL`
3. Set number of articles: `10`
4. Click **"🔍 Analyze Sentiment"**

**What to show:**
- Overall sentiment score
- Positive/Neutral/Negative breakdown
- Sentiment distribution pie chart
- Detailed article list with scores

**Expected time:** 30 seconds

---

### Demo 4: Portfolio Optimization

**Purpose:** Optimize portfolio using Modern Portfolio Theory

**Steps:**
1. Click **"Portfolio Optimization"** in the sidebar
2. Enter stock symbols: `AAPL,MSFT,GOOGL,AMZN,TSLA`
3. Select data period: `1y`
4. Choose optimization method: `Maximize Sharpe Ratio`
5. Click **"⚡ Optimize Portfolio"**

**What to show:**
- Expected return, volatility, Sharpe ratio
- Optimal weight allocation table
- Weight distribution pie chart
- Efficient frontier scatter plot
- Method comparison table

**Expected time:** 1 minute

---

### Demo 5: Market Analysis

**Purpose:** Technical analysis with indicators

**Steps:**
1. Click **"Market Analysis"** in the sidebar
2. Enter stock symbol: `AAPL`
3. Select period: `1y`
4. Click **"Load Market Data"**

**What to show:**
- Current price, market cap, P/E ratio
- Candlestick price chart with moving averages
- RSI indicator (overbought/oversold)
- MACD indicator
- Volume analysis

**Expected time:** 30 seconds

---

## Presentation Tips

### For Business Audiences

**Focus on:**
1. **Cross-Sectional Alpha Engine** (most impressive)
   - "This is how top quant funds rank stocks"
   - "Risk adjustment is the key differentiator"
   - "Output is portfolio weights, not predictions"

2. **User Interface**
   - "Professional, enterprise-grade design"
   - "No coding required - point and click"
   - "Real-time data integration"

3. **Practical Applications**
   - "Portfolio construction"
   - "Risk management"
   - "Research and backtesting"

### For Technical Audiences

**Focus on:**
1. **Architecture**
   - Three-layer design (Alpha → Risk → Allocation)
   - Modular, extensible codebase
   - Industry-standard practices

2. **Methodology**
   - Cross-sectional vs time-series
   - Risk adjustment necessity
   - Evaluation metrics (Rank IC)

3. **Code Quality**
   - Type hints
   - Comprehensive docstrings
   - Error handling

---

## Troubleshooting

### Problem: "Module not found" error

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "API rate limit exceeded"

**Solution:**
- Wait 1-2 minutes and try again
- Or get your own API key (free) from Alpha Vantage

### Problem: Browser doesn't open automatically

**Solution:**
- Manually open: `http://localhost:8501`
- Check Terminal for the exact URL

### Problem: "Insufficient data" error

**Solution:**
- Select a longer data period (2y or 5y)
- Try a different stock symbol
- Ensure internet connection is working

### Problem: Application won't start

**Solution:**
1. Check Python version: `python --version` (should be 3.9+)
2. Reinstall dependencies: `pip install -r requirements.txt --upgrade`
3. Check for error messages in Terminal

### Problem: Slow performance

**Solution:**
- Use shorter data periods for demos (1y instead of 5y)
- Reduce number of stocks in cross-sectional analysis
- Close other applications to free up memory

---

## Quick Reference

### Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start application
streamlit run app.py

# Stop application
Ctrl + C
```

### Key URLs

- Application: `http://localhost:8501`
- Alpha Vantage API: `https://www.alphavantage.co/support/#api-key`

### Recommended Demo Flow

1. **Start with Market Analysis** (quick, visual)
2. **Show Sentiment Analysis** (easy to understand)
3. **Demonstrate Portfolio Optimization** (practical)
4. **Highlight Cross-Sectional Alpha Engine** (key differentiator)
5. **End with LSTM Prediction** (impressive but takes time)

**Total demo time:** 10-15 minutes

---

## Support

For additional help:
- Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Review [README.md](README.md) for technical details
- Open an issue on GitHub

---

**Last Updated:** 2024

**Version:** 1.0

