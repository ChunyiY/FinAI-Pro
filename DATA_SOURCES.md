# 📡 Data Sources Guide

## Overview

The application now supports **multiple data sources** with automatic failover to ensure reliable data access even when one API is unavailable or rate-limited.

## Supported Data Sources

### 1. **yfinance** (Default)
- **Source**: Yahoo Finance via yfinance library
- **Pros**: Free, no API key needed, comprehensive data
- **Cons**: Rate limits, occasional downtime
- **Best for**: General use, most stocks

### 2. **pandas_datareader**
- **Source**: Yahoo Finance via pandas_datareader
- **Pros**: Alternative Yahoo Finance backend, free
- **Cons**: Same rate limits as yfinance
- **Best for**: Backup when yfinance fails

### 3. **Demo Data** (Fallback)
- **Source**: Synthetic data generator
- **Pros**: No API calls, unlimited, always works
- **Cons**: Not real data
- **Best for**: Demos, testing, offline use

## How It Works

### Automatic Failover

The system tries data sources in this order:

1. **Preferred source** (selected in sidebar)
2. **yfinance** (if not preferred)
3. **pandas_datareader** (if not preferred)
4. **Demo data** (if all APIs fail)

### Example Flow

```
Request for AAPL data
  ↓
Try yfinance → Rate limited ❌
  ↓
Try pandas_datareader → Success ✅
  ↓
Return data
```

## Usage

### Method 1: Auto Mode (Recommended)

1. Open the application
2. In sidebar, select **"Auto (Failover)"** for Data Source
3. System automatically tries sources in order
4. No manual intervention needed

### Method 2: Manual Selection

1. Select specific source: **"yfinance"** or **"pandas_datareader"**
2. System uses only that source
3. Falls back to demo data if it fails

### Method 3: Demo Mode

1. Enable **"Demo Mode"** checkbox
2. Uses only synthetic data
3. No API calls at all

## Benefits

### ✅ Reliability

- **No single point of failure** - Multiple sources
- **Automatic failover** - Seamless switching
- **Always works** - Demo data as final fallback

### ✅ Performance

- **Fast failover** - Quick source switching
- **Smart caching** - Reduces API calls
- **Efficient** - Only tries necessary sources

### ✅ Flexibility

- **User choice** - Select preferred source
- **Auto mode** - Let system decide
- **Demo mode** - For offline/testing

## Technical Details

### Source Priority

When in Auto mode, sources are tried in this order:

1. **yfinance** (if available)
2. **pandas_datareader** (backup)
3. **Demo data** (final fallback)

### Error Handling

- **Rate limits**: Quickly skip to next source
- **Timeouts**: Try next source after short delay
- **Network errors**: Immediate failover
- **Empty data**: Try next source

### Caching

- Data is cached per symbol/period/interval
- Cache persists during session
- Reduces API calls across all sources

## Comparison

| Feature | yfinance | pandas_datareader | Demo Data |
|---------|----------|-------------------|-----------|
| **Free** | ✅ | ✅ | ✅ |
| **API Key** | ❌ | ❌ | ❌ |
| **Rate Limits** | ⚠️ | ⚠️ | ❌ |
| **Real Data** | ✅ | ✅ | ❌ |
| **Reliability** | Medium | Medium | High |
| **Speed** | Fast | Fast | Instant |

## Troubleshooting

### All Sources Failing

**Issue**: All APIs return errors

**Solution**:
- Enable **Demo Mode** for guaranteed data
- Check internet connection
- Wait a few minutes (rate limits reset)

### Slow Data Loading

**Issue**: Data takes too long to load

**Solution**:
- Enable **Demo Mode** for instant data
- Use cached data (try same symbol again)
- Check network connection

### Rate Limit Errors

**Issue**: Still seeing rate limit errors

**Solution**:
- System automatically tries other sources
- Enable **Demo Mode** to avoid API calls
- Wait 2-5 minutes for limits to reset

## Best Practices

### For Production Use

1. Use **Auto (Failover)** mode
2. Monitor which source is being used
3. Consider adding paid APIs for higher limits

### For Demos/Interviews

1. Enable **Demo Mode** before starting
2. No API failures during presentation
3. Consistent, reliable data

### For Development

1. Use **Demo Mode** for testing
2. Test failover with **Auto mode**
3. Verify all sources work

## Future Enhancements

Potential additional data sources:

- **Alpha Vantage** (requires free API key)
- **Polygon.io** (requires API key)
- **IEX Cloud** (requires API key)
- **Quandl** (requires API key)

## Summary

The multi-source data fetcher provides:

- ✅ **Reliability** - Multiple sources with failover
- ✅ **Flexibility** - Choose your preferred source
- ✅ **Resilience** - Always works with demo fallback
- ✅ **Performance** - Smart caching and fast failover

**Use Auto mode for best experience!** 🚀

