# 📊 Alpha Vantage Only Configuration

## Overview

This application now uses **Alpha Vantage API exclusively** as the data source. All other data sources have been removed.

## API Key Configuration

Your Alpha Vantage API key: `LODPOAHFH4DIACRR`

This key is configured in `config.py` and will be automatically used.

## Alpha Vantage API Details

### Endpoint Used
- **Function**: `TIME_SERIES_DAILY`
- **URL**: `https://www.alphavantage.co/query`
- **Output Size**: `compact` (free tier - returns last ~100 data points)

### Free Tier Limitations
- **Rate Limit**: 5 API calls per minute
- **Data Points**: ~100 days of historical data (compact mode)
- **Full History**: Requires premium subscription

### Rate Limiting
The application automatically:
- Waits 12 seconds between API calls
- Handles rate limit errors gracefully
- Shows clear error messages if rate limit is exceeded

## Features

### ✅ What Works
- Daily stock price data (OHLCV)
- Last ~100 trading days of data
- Technical indicators (MA, RSI, MACD, Bollinger Bands)
- Stock information (company overview)
- Multiple stocks support

### ⚠️ Limitations (Free Tier)
- Limited to ~100 days of historical data
- 5 calls per minute rate limit
- No intraday data
- No full history (requires premium)

## Usage

### Basic Usage
```python
from real_data_fetcher import RealDataFetcher

fetcher = RealDataFetcher()
data = fetcher.get_stock_data('AAPL', period='1y')
```

### Rate Limit Handling
The system automatically waits between calls. If you see rate limit errors:
- Wait 1 minute before trying again
- The system will automatically retry with delays

## API Response Format

Alpha Vantage returns data in this format:
```json
{
  "Time Series (Daily)": {
    "2024-01-15": {
      "1. open": "150.00",
      "2. high": "152.00",
      "3. low": "149.00",
      "4. close": "151.00",
      "5. volume": "1000000"
    }
  }
}
```

The application automatically converts this to a pandas DataFrame with standard column names.

## Error Handling

### Common Errors

1. **Rate Limit Exceeded**
   - Error: "Note: Thank you for using Alpha Vantage..."
   - Solution: Wait 1 minute, system will retry automatically

2. **Invalid Symbol**
   - Error: "Error Message" in response
   - Solution: Check symbol is correct (e.g., 'AAPL' not 'apple')

3. **API Key Invalid**
   - Error: "Invalid API key"
   - Solution: Check `config.py` has correct API key

## Premium Features

To unlock premium features:
- Visit: https://www.alphavantage.co/premium/
- Premium tiers offer:
  - Full historical data (20+ years)
  - Higher rate limits
  - Intraday data
  - Real-time quotes

## Summary

✅ **Single Data Source**: Alpha Vantage only
✅ **API Key Configured**: Already set in config.py
✅ **Rate Limiting**: Automatic handling
✅ **Error Handling**: Comprehensive error messages
⚠️ **Free Tier Limits**: ~100 days, 5 calls/minute

**The application is ready to use with Alpha Vantage!** 🚀

