# 📈 Real Data Sources Guide

## Overview

The application now uses **Real Data Fetcher** which supports **multiple real market data sources**. No demo/synthetic data - only real stock market data!

## Supported Real Data Sources

### 1. **yfinance** (Yahoo Finance) ✅
- **Free**: Yes, no API key needed
- **Rate Limit**: Yes (varies)
- **Coverage**: Global stocks
- **Install**: `pip install yfinance`

### 2. **pandas_datareader** (Yahoo Finance Alternative) ✅
- **Free**: Yes, no API key needed
- **Rate Limit**: Yes (same as yfinance)
- **Coverage**: Global stocks
- **Install**: `pip install pandas-datareader`

### 3. **Alpha Vantage** 🔑
- **Free**: Yes, free API key required
- **Rate Limit**: 5 calls/minute (free tier)
- **Coverage**: Global stocks, forex, crypto
- **Get API Key**: https://www.alphavantage.co/support/#api-key
- **Setup**: `export ALPHA_VANTAGE_API_KEY=your_key`

### 4. **Finnhub** 🔑
- **Free**: Yes, free API key required
- **Rate Limit**: 60 calls/minute (free tier)
- **Coverage**: Global stocks, crypto, news
- **Get API Key**: https://finnhub.io/register
- **Setup**: `export FINNHUB_API_KEY=your_key`

### 5. **Polygon.io** 🔑
- **Free**: Yes, free API key required
- **Rate Limit**: 5 calls/minute (free tier)
- **Coverage**: US stocks, options, crypto
- **Get API Key**: https://polygon.io/
- **Setup**: `export POLYGON_API_KEY=your_key`

### 6. **IEX Cloud** 🔑
- **Free**: Yes, free API key required
- **Rate Limit**: 50,000 messages/month (free tier)
- **Coverage**: US stocks
- **Get API Key**: https://iexcloud.io/
- **Setup**: `export IEX_API_KEY=your_key`

## How It Works

### Automatic Failover

The system tries sources in this order:

1. **yfinance** (if available)
2. **pandas_datareader** (if available)
3. **Alpha Vantage** (if API key set)
4. **Finnhub** (if API key set)
5. **Polygon.io** (if API key set)
6. **IEX Cloud** (if API key set)

### Example Flow

```
Request for AAPL data
  ↓
Try yfinance → Rate limited ❌
  ↓
Try pandas_datareader → Rate limited ❌
  ↓
Try Alpha Vantage → Success ✅
  ↓
Return real market data
```

## Setup Instructions

### Basic Setup (No API Keys)

```bash
# Install basic sources
pip install yfinance pandas-datareader

# That's it! You have 2 real data sources
```

### Enhanced Setup (With API Keys)

#### Step 1: Get Free API Keys

1. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
2. **Finnhub**: https://finnhub.io/register
3. **Polygon.io**: https://polygon.io/
4. **IEX Cloud**: https://iexcloud.io/

#### Step 2: Set Environment Variables

**On macOS/Linux:**
```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
export FINNHUB_API_KEY=your_key_here
export POLYGON_API_KEY=your_key_here
export IEX_API_KEY=your_key_here
```

**On Windows:**
```cmd
set ALPHA_VANTAGE_API_KEY=your_key_here
set FINNHUB_API_KEY=your_key_here
set POLYGON_API_KEY=your_key_here
set IEX_API_KEY=your_key_here
```

**Or create a `.env` file:**
```
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
IEX_API_KEY=your_key_here
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### No Configuration Needed!

Just start the application:

```bash
streamlit run app.py
```

The system will:
- ✅ Automatically detect available sources
- ✅ Show available sources in sidebar
- ✅ Try sources in order
- ✅ Use real market data only

### Sidebar Display

The sidebar will show:
```
📡 Real data sources:
yfinance, pandas_datareader, alpha_vantage

✅ Using real market data only
System automatically tries sources in order
```

## Benefits

### ✅ Multiple Real Data Sources

- **6 different sources** - More than enough backup
- **Automatic failover** - Seamless switching
- **Real market data** - No synthetic data

### ✅ Reliability

- **No single point of failure** - Multiple sources
- **Rate limit handling** - Automatically switches
- **Always real data** - Never falls back to demo

### ✅ Flexibility

- **Works with or without API keys** - Basic sources work immediately
- **Add more sources** - Just set API keys
- **Smart caching** - Reduces API calls

## Rate Limits Comparison

| Source | Free Tier Rate Limit |
|--------|---------------------|
| yfinance | Variable (Yahoo limits) |
| pandas_datareader | Variable (Yahoo limits) |
| Alpha Vantage | 5 calls/minute |
| Finnhub | 60 calls/minute |
| Polygon.io | 5 calls/minute |
| IEX Cloud | 50,000 messages/month |

## Best Practices

### For Maximum Reliability

1. **Install basic sources**: `pip install yfinance pandas-datareader`
2. **Get at least 2 API keys**: Alpha Vantage + Finnhub recommended
3. **Set environment variables**: So system can use them
4. **Let system handle failover**: Automatic switching

### For High Frequency Use

1. **Get multiple API keys**: More sources = more calls
2. **Use caching**: System caches automatically
3. **Monitor rate limits**: System handles automatically
4. **Consider paid tiers**: For production use

## Troubleshooting

### No Sources Available

**Issue**: All sources show as unavailable

**Solution**:
```bash
# Install at least basic sources
pip install yfinance pandas-datareader

# Restart application
```

### API Keys Not Working

**Issue**: API key sources not detected

**Solution**:
- Check environment variables: `echo $ALPHA_VANTAGE_API_KEY`
- Verify API key is correct
- Check API key hasn't expired
- Restart application after setting variables

### Still Getting Rate Limits

**Issue**: All sources rate limited

**Solution**:
- System automatically tries next source
- Wait a few minutes for limits to reset
- Get more API keys for more sources
- Use caching (system does this automatically)

## Summary

The Real Data Fetcher provides:

- ✅ **6 real data sources** - More than enough
- ✅ **Automatic failover** - Seamless switching
- ✅ **Real market data only** - No synthetic data
- ✅ **Works with or without API keys** - Basic setup works immediately
- ✅ **Smart caching** - Reduces API calls

**Install basic sources and optionally add API keys for maximum reliability!** 🚀

