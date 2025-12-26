# 🎭 Demo Mode Guide

## Overview

Demo Mode allows you to use synthetic stock data instead of the Yahoo Finance API. This is perfect for:
- **Avoiding rate limits** - No API calls needed
- **Offline demonstrations** - Works without internet
- **Consistent testing** - Same data every time
- **Interview demos** - Reliable and fast

## How to Enable Demo Mode

### Method 1: Sidebar Toggle (Recommended)

1. Open the Streamlit application
2. Look for the sidebar checkbox: **"🎭 Demo Mode (Use synthetic data)"**
3. Check the box to enable demo mode
4. All data will now come from synthetic data generator

### Method 2: Automatic Fallback

The system automatically falls back to demo data when:
- API rate limits are exceeded
- Network timeouts occur
- API is unavailable

You'll see a message: `⚠️ API unavailable. Using demo data for SYMBOL...`

## Supported Stock Symbols

Demo mode includes realistic templates for popular stocks:

- **AAPL** (Apple) - Base price: $175
- **MSFT** (Microsoft) - Base price: $380
- **GOOGL** (Google) - Base price: $140
- **AMZN** (Amazon) - Base price: $150
- **TSLA** (Tesla) - Base price: $250
- **META** (Meta/Facebook) - Base price: $320
- **NVDA** (NVIDIA) - Base price: $450

**Note**: Other symbols will use default parameters (base price: $100)

## Features

### Realistic Data Generation

Demo data includes:
- ✅ OHLCV (Open, High, Low, Close, Volume) data
- ✅ Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
- ✅ Realistic price movements using geometric Brownian motion
- ✅ Volume correlated with price changes
- ✅ Valid OHLC relationships

### Data Characteristics

- **Price movements**: Based on geometric Brownian motion with drift
- **Volatility**: Realistic volatility per stock
- **Trends**: Small positive trends for most stocks
- **Volume**: Higher volume on larger price changes

## Usage Examples

### Stock Prediction with Demo Data

1. Enable Demo Mode in sidebar
2. Go to "Stock Prediction" page
3. Enter any stock symbol (AAPL, MSFT, etc.)
4. Click "Start Prediction"
5. Data loads instantly without API calls!

### Portfolio Optimization with Demo Data

1. Enable Demo Mode
2. Go to "Portfolio Optimization" page
3. Enter multiple symbols: `AAPL,MSFT,GOOGL`
4. All data comes from demo generator
5. No rate limits!

## Technical Details

### Data Generation Algorithm

The demo data generator uses:
- **Geometric Brownian Motion**: For realistic price paths
- **Random walk with drift**: For trend simulation
- **Volatility modeling**: Stock-specific volatility
- **Volume correlation**: Volume increases with price volatility

### Code Structure

```
demo_data.py
├── DemoDataGenerator class
│   ├── generate_stock_data() - Main data generation
│   ├── get_stock_info() - Synthetic stock info
│   └── _add_technical_indicators() - Calculate indicators
```

## Advantages

### ✅ Benefits

1. **No Rate Limits** - Unlimited requests
2. **Fast** - Instant data generation
3. **Reliable** - Always works
4. **Consistent** - Same seed = same data
5. **Offline** - No internet needed
6. **Perfect for Demos** - No API failures

### ⚠️ Limitations

1. **Not Real Data** - Synthetic data only
2. **Limited Symbols** - Best results with predefined symbols
3. **No News Data** - Sentiment analysis still needs API
4. **No Real-time Updates** - Data is generated, not live

## Best Practices

### For Interviews/Demos

1. **Enable Demo Mode** before starting
2. **Test with popular symbols** (AAPL, MSFT, GOOGL)
3. **Explain the feature** - Shows robust error handling
4. **Show both modes** - Demonstrate API fallback

### For Development

1. **Use Demo Mode** for testing
2. **Test with various symbols**
3. **Verify technical indicators**
4. **Check data consistency**

## Troubleshooting

### Demo Mode Not Working

**Issue**: Checkbox doesn't enable demo mode

**Solution**: 
- Refresh the page
- Check that `demo_data.py` exists
- Verify imports are working

### Data Looks Unrealistic

**Issue**: Generated data seems off

**Solution**:
- Use predefined symbols (AAPL, MSFT, etc.)
- Adjust period (1y works best)
- Data is synthetic, so some variation is expected

### Still Getting Rate Limit Errors

**Issue**: Even with demo mode enabled

**Solution**:
- Make sure checkbox is checked
- Check sidebar for demo mode indicator
- Sentiment analysis still uses API (news data)

## Comparison: API vs Demo Mode

| Feature | API Mode | Demo Mode |
|---------|----------|-----------|
| Data Source | Yahoo Finance | Synthetic Generator |
| Rate Limits | Yes | No |
| Internet Required | Yes | No |
| Real Data | Yes | No |
| Speed | Variable | Instant |
| Reliability | Depends on API | Always works |
| Best For | Production | Demos/Testing |

## Code Example

```python
from demo_data import DemoDataGenerator

# Create generator
generator = DemoDataGenerator(seed=42)

# Generate data for AAPL
data = generator.generate_stock_data('AAPL', period='1y', interval='1d')

# Get stock info
info = generator.get_stock_info('AAPL')
```

## Summary

Demo Mode is a powerful feature that:
- ✅ Eliminates rate limit issues
- ✅ Provides reliable demo data
- ✅ Works offline
- ✅ Perfect for interviews and testing

**Enable it in the sidebar and enjoy unlimited, instant data!** 🚀

