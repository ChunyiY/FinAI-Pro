# 🧠 Smart Data Fetcher Guide

## Overview

The **Smart Data Fetcher** automatically detects which data sources are available on your system and only uses those that actually work. No more guessing or manual configuration!

## How It Works

### Automatic Detection

When the application starts, the Smart Data Fetcher:

1. **Tests each data source** - Checks if modules can be imported
2. **Creates a list** - Only includes sources that are available
3. **Uses them in order** - Tries available sources automatically
4. **Fails gracefully** - Falls back to demo data if all fail

### Detection Process

```
Start Application
  ↓
Test yfinance → Available? Add to list
  ↓
Test pandas_datareader → Available? Add to list
  ↓
Test demo_data → Available? Add to list
  ↓
Use available sources in order
```

## What You'll See

When the app starts, you'll see output like:

```
============================================================
Initializing Smart Data Fetcher...
============================================================
✓ yfinance detected and available
✗ pandas_datareader not available (not installed)
✓ demo_data detected and available

📊 Available sources (in order): yfinance, demo_data
============================================================
```

## Benefits

### ✅ Automatic Configuration

- **No manual setup** - Detects what's installed
- **No errors** - Only uses available sources
- **Smart fallback** - Always has a working source

### ✅ Dynamic Adaptation

- **Installs new source?** - Automatically detects it
- **Source fails?** - Tries next one automatically
- **Always works** - Demo data as final fallback

### ✅ User-Friendly

- **Clear feedback** - Shows what's available
- **No confusion** - Only shows working sources
- **Reliable** - Never fails completely

## Example Scenarios

### Scenario 1: Only yfinance Installed

```
Available sources: yfinance, demo_data
Order: yfinance → demo_data
```

### Scenario 2: All Sources Installed

```
Available sources: yfinance, pandas_datareader, demo_data
Order: yfinance → pandas_datareader → demo_data
```

### Scenario 3: Only Demo Data Available

```
Available sources: demo_data
Order: demo_data (always works!)
```

## Usage

### No Configuration Needed!

Just start the application:

```bash
streamlit run app.py
```

The Smart Data Fetcher will:
- ✅ Detect available sources
- ✅ Show what's available in sidebar
- ✅ Use them automatically
- ✅ Handle failures gracefully

### Sidebar Display

The sidebar will show:
```
📡 Available sources:
yfinance, demo_data

System automatically uses available sources in order
```

## Testing Your Setup

To see what sources are available, check the console output when starting the app, or look at the sidebar.

You can also run the test script:

```bash
python test_sources_simple.py
```

## Installation Recommendations

### Minimum Setup (Always Works)

```bash
# Demo data is built-in, always works
# No installation needed!
```

### Recommended Setup

```bash
pip install yfinance
# Now you have: yfinance + demo_data
```

### Full Setup

```bash
pip install yfinance pandas-datareader
# Now you have: yfinance + pandas_datareader + demo_data
```

## How It Handles Failures

### Rate Limits

If yfinance is rate limited:
1. Tries pandas_datareader (if available)
2. Falls back to demo_data (always available)

### Network Errors

If network fails:
1. Tries next available source
2. Falls back to demo_data

### Module Not Installed

If module not installed:
- Simply not included in available sources
- System uses what's available

## Comparison with Other Fetchers

| Feature | Smart Fetcher | Multi-Source | Basic Fetcher |
|---------|---------------|--------------|---------------|
| **Auto Detection** | ✅ | ❌ | ❌ |
| **Only Uses Available** | ✅ | ❌ | ❌ |
| **No Configuration** | ✅ | ⚠️ | ✅ |
| **Smart Fallback** | ✅ | ✅ | ⚠️ |
| **User Feedback** | ✅ | ⚠️ | ❌ |

## Technical Details

### Detection Method

The fetcher tests imports, not API calls:
- Fast detection (no network calls)
- No rate limits during detection
- Reliable (doesn't depend on API status)

### Source Priority

1. **yfinance** (if available) - Most reliable
2. **pandas_datareader** (if available) - Alternative Yahoo backend
3. **demo_data** (always available) - Guaranteed fallback

### Caching

- Data cached per symbol/period/interval
- Reduces API calls
- Works across all sources

## Troubleshooting

### No Sources Available

**Issue**: All sources show as unavailable

**Solution**:
```bash
# Install at least one source
pip install yfinance

# Or use demo data (always works)
# Demo mode is always available
```

### Source Not Detected

**Issue**: Installed but not detected

**Solution**:
- Restart the application
- Check if module imports correctly: `python -c "import yfinance"`
- Verify installation: `pip list | grep yfinance`

### Still Getting Errors

**Issue**: Source detected but fails at runtime

**Solution**:
- System automatically tries next source
- Check network connection
- Enable demo mode for guaranteed data

## Summary

The Smart Data Fetcher:

- ✅ **Automatically detects** available sources
- ✅ **Only uses** what's actually installed
- ✅ **No configuration** needed
- ✅ **Always works** with demo fallback
- ✅ **User-friendly** with clear feedback

**Just install what you want and let the system handle the rest!** 🚀

