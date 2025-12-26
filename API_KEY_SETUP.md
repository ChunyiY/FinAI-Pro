# 🔑 API Key Setup Guide

## Alpha Vantage API Key

Your Alpha Vantage API key has been configured: `LODPOAHFH4DIACRR`

### Method 1: Using Config File (Already Done) ✅

The API key is already set in `config.py`. The system will automatically use it.

### Method 2: Environment Variable (Optional)

You can also set it as an environment variable:

**macOS/Linux:**
```bash
export ALPHA_VANTAGE_API_KEY=LODPOAHFH4DIACRR
```

**Windows:**
```cmd
set ALPHA_VANTAGE_API_KEY=LODPOAHFH4DIACRR
```

### Method 3: Using Setup Script

Run the setup script:
```bash
source setup_api_keys.sh
```

## Current Data Sources

With your Alpha Vantage API key configured, you now have:

### FREE Sources (No API Key):
1. ✅ **yfinance** - Yahoo Finance
2. ✅ **pandas_datareader** - Yahoo Finance alternative
3. ✅ **investpy** - Investing.com
4. ✅ **yahooquery** - Yahoo Finance alternative library

### With API Key:
5. ✅ **Alpha Vantage** - Now available! (API key: LODPOAHFH4DIACRR)

**Total: 5 real data sources!**

## Verification

When you start the application, you should see:
```
✓ alpha_vantage - Available (API key configured)
```

And in the sidebar:
```
📡 Real data sources:
yfinance, pandas_datareader, investpy, yahooquery, alpha_vantage
```

## Data Source Priority

The system will try sources in this order:

1. yfinance
2. pandas_datareader
3. investpy
4. yahooquery
5. **alpha_vantage** ← Your API key source
6. Other API key sources (if configured)

## Alpha Vantage Rate Limits

- **Free Tier**: 5 API calls per minute
- **500 API calls per day**

The system automatically handles rate limits and switches to other sources.

## Notes

- The API key is stored in `config.py` (already configured)
- The key is also in `.env.example` for reference
- Never commit `.env` file to git (it's in .gitignore)
- The system automatically detects and uses the API key

## Troubleshooting

If Alpha Vantage is not detected:

1. Check that `config.py` exists and has the API key
2. Restart the application
3. Check console output for detection messages

Your API key is ready to use! 🚀

