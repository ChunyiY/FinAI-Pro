# 🚀 Quick Setup Guide

## Your Alpha Vantage API Key

✅ **API Key**: `LODPOAHFH4DIACRR`

This key has been configured in `config.py` and will be automatically used by the system.

## Current Data Sources

You now have **5 real data sources**:

### FREE Sources (No API Key):
1. ✅ **yfinance** - Yahoo Finance
2. ✅ **pandas_datareader** - Yahoo Finance alternative  
3. ✅ **investpy** - Investing.com
4. ✅ **yahooquery** - Yahoo Finance alternative library

### With Your API Key:
5. ✅ **Alpha Vantage** - Configured with your API key!

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Start Application

```bash
streamlit run app.py
```

## What You'll See

When the app starts, you should see:

```
============================================================
Initializing Real Data Fetcher (Real Market Data Only)...
============================================================
✓ yfinance (Yahoo Finance) - Available - NO API KEY NEEDED
✓ pandas_datareader (Yahoo Finance) - Available - NO API KEY NEEDED
✓ investpy (Investing.com) - Available - NO API KEY NEEDED
✓ yahooquery (Yahoo Finance) - Available - NO API KEY NEEDED
✓ alpha_vantage - Available (API key configured)

📊 Real data sources available: yfinance, pandas_datareader, investpy, yahooquery, alpha_vantage
✅ FREE sources (no API key): yfinance, pandas_datareader, investpy, yahooquery
📈 Total free sources: 4
============================================================
```

## Data Source Priority

The system tries sources in this order:

1. **yfinance** (Yahoo Finance)
2. **pandas_datareader** (Yahoo Finance alternative)
3. **investpy** (Investing.com)
4. **yahooquery** (Yahoo Finance alternative)
5. **alpha_vantage** (Your API key source) ← Extra backup!

## Alpha Vantage Rate Limits

- **5 API calls per minute**
- **500 API calls per day**

The system automatically handles rate limits and switches to other sources.

## Summary

✅ **5 real data sources** configured
✅ **Alpha Vantage API key** ready to use
✅ **No additional setup** needed
✅ **Automatic failover** between sources

**You're all set!** 🎉

