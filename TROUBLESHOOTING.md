# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 1. Rate Limit Exceeded (Too Many Requests)

**Error Message:**
```
Error fetching data for AAPL: Too Many Requests. Rate limited. Try after a while.
```

**What it means:**
Yahoo Finance API limits the number of requests you can make in a short period. This is a protective measure to prevent abuse.

**Solutions:**

1. **Wait 2-5 minutes** - Rate limits are temporary and reset automatically
2. **Use cached data** - The application caches previously fetched data
3. **Reduce request frequency** - Wait a few seconds between requests
4. **Automatic retry** - The system automatically retries with exponential backoff (4s, 8s, 16s delays)

**How the system handles it:**
- Automatically detects rate limit errors
- Retries up to 3 times with increasing delays
- Shows friendly error message with solutions
- Uses exponential backoff (longer waits for rate limits)

### 2. Network Timeout

**Error Message:**
```
Connection timed out after 30002 milliseconds
```

**What it means:**
The request took too long to complete, likely due to network issues or API unavailability.

**Solutions:**

1. **Check internet connection** - Ensure you have stable internet
2. **Wait and retry** - Network issues are often temporary
3. **Try different stock** - Some symbols may have data availability issues
4. **Automatic retry** - System retries with 2s, 4s, 6s delays

### 3. Module Not Found Errors

**Error Message:**
```
ModuleNotFoundError: No module named 'nltk'
```

**Solution:**
```bash
pip install -r requirements.txt
python setup.py
```

### 4. Empty Data Returned

**Error Message:**
```
Unable to fetch data for SYMBOL. Please check the symbol.
```

**Possible causes:**
- Invalid stock symbol
- Market is closed
- Symbol doesn't exist or has been delisted

**Solutions:**
- Verify the symbol is correct (e.g., AAPL not AAPL.US)
- Try popular symbols: AAPL, MSFT, GOOGL, TSLA, AMZN
- Check if market is open (for real-time data)

## Best Practices

### To Avoid Rate Limits:

1. **Don't spam requests** - Wait a few seconds between different stock queries
2. **Use cached data** - The app caches data to reduce API calls
3. **Batch operations** - When optimizing portfolios, the system fetches multiple stocks efficiently
4. **Test with popular symbols** - AAPL, MSFT, GOOGL are reliable

### For Demo/Interview:

1. **Prepare test data** - Fetch data once before the demo and let it cache
2. **Use the same symbols** - Stick to symbols you've tested
3. **Have backup plan** - If API fails, explain the error handling features
4. **Show error handling** - Rate limit errors demonstrate robust error handling

## Technical Details

### Retry Mechanism

The system implements intelligent retry logic:

- **Rate Limits**: Exponential backoff (4s, 8s, 16s)
- **Network Errors**: Linear backoff (2s, 4s, 6s)
- **Other Errors**: Immediate failure (no retry)

### Caching

- Data is cached per symbol/period/interval combination
- Cache persists during the session
- Reduces API calls and improves performance

### Error Detection

The system automatically detects:
- Rate limiting errors
- Network timeouts
- Connection errors
- Invalid symbols
- Empty data responses

## Getting Help

If you continue to experience issues:

1. Check your internet connection
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Try running `python verify_code.py` to check code syntax
4. Check the terminal output for detailed error messages
5. Wait a few minutes if rate limited, then try again

## Rate Limit Information

Yahoo Finance API rate limits are:
- Not publicly documented
- Typically reset within 2-5 minutes
- More lenient for individual users vs. automated scripts
- May vary by region and time of day

**Note**: This is a free API service, so rate limits are expected. For production use, consider:
- Using paid financial data APIs
- Implementing request queuing
- Using multiple API keys (if available)
- Caching data more aggressively

