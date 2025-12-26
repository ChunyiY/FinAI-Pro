"""
Simple test to check which data sources are actually available.
"""
import sys

def check_module(module_name, import_name=None):
    """Check if a module can be imported."""
    if import_name is None:
        import_name = module_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

print("Checking data source availability...")
print("=" * 50)

# Check yfinance
yfinance_ok = check_module('yfinance')
print(f"yfinance: {'✓ Available' if yfinance_ok else '✗ Not available'}")

# Check pandas_datareader
pandas_dr_ok = check_module('pandas_datareader')
print(f"pandas_datareader: {'✓ Available' if pandas_dr_ok else '✗ Not available'}")

# Check demo_data (local file)
try:
    import demo_data
    demo_ok = True
except ImportError:
    demo_ok = False
print(f"demo_data: {'✓ Available' if demo_ok else '✗ Not available'}")

print("=" * 50)
available = []
if yfinance_ok:
    available.append('yfinance')
if pandas_dr_ok:
    available.append('pandas_datareader')
if demo_ok:
    available.append('demo_data')

if available:
    print(f"\nAvailable sources: {', '.join(available)}")
    print(f"\nRecommendation: Use {' -> '.join(available)} in order")
else:
    print("\n⚠️  No sources available! Please install dependencies:")
    print("   pip install yfinance pandas-datareader")

