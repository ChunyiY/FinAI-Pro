"""
Test Data Sources Availability

Tests which data sources are actually accessible and working.
"""
import sys
from datetime import datetime, timedelta

def test_yfinance():
    """Test yfinance data source."""
    print("=" * 60)
    print("Testing yfinance...")
    print("-" * 60)
    try:
        import yfinance as yf
        print("✓ yfinance module imported successfully")
        
        # Test fetching data for AAPL
        print("Testing data fetch for AAPL...")
        ticker = yf.Ticker("AAPL")
        df = ticker.history(period="5d")
        
        if df.empty:
            print("✗ yfinance: Data fetch returned empty DataFrame")
            return False
        
        print(f"✓ yfinance: Successfully fetched {len(df)} rows")
        print(f"  Latest price: ${df['Close'].iloc[-1]:.2f}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        return True
        
    except ImportError:
        print("✗ yfinance: Module not installed")
        return False
    except Exception as e:
        error_msg = str(e).lower()
        if 'rate limit' in error_msg or 'too many requests' in error_msg:
            print(f"✗ yfinance: Rate limited - {str(e)[:100]}")
        elif 'timeout' in error_msg or 'connection' in error_msg:
            print(f"✗ yfinance: Network error - {str(e)[:100]}")
        else:
            print(f"✗ yfinance: Error - {str(e)[:100]}")
        return False

def test_pandas_datareader():
    """Test pandas_datareader data source."""
    print("\n" + "=" * 60)
    print("Testing pandas_datareader...")
    print("-" * 60)
    try:
        import pandas_datareader.data as web
        print("✓ pandas_datareader module imported successfully")
        
        # Test fetching data for AAPL
        print("Testing data fetch for AAPL...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        df = web.get_data_yahoo("AAPL", start=start_date, end=end_date)
        
        if df.empty:
            print("✗ pandas_datareader: Data fetch returned empty DataFrame")
            return False
        
        print(f"✓ pandas_datareader: Successfully fetched {len(df)} rows")
        print(f"  Latest price: ${df['Close'].iloc[-1]:.2f}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        return True
        
    except ImportError:
        print("✗ pandas_datareader: Module not installed")
        print("  Install with: pip install pandas-datareader")
        return False
    except Exception as e:
        error_msg = str(e).lower()
        if 'rate limit' in error_msg or 'too many requests' in error_msg:
            print(f"✗ pandas_datareader: Rate limited - {str(e)[:100]}")
        elif 'timeout' in error_msg or 'connection' in error_msg:
            print(f"✗ pandas_datareader: Network error - {str(e)[:100]}")
        else:
            print(f"✗ pandas_datareader: Error - {str(e)[:100]}")
        return False

def test_demo_data():
    """Test demo data generator."""
    print("\n" + "=" * 60)
    print("Testing demo data generator...")
    print("-" * 60)
    try:
        from demo_data import DemoDataGenerator
        print("✓ demo_data module imported successfully")
        
        # Test generating data
        print("Testing data generation for AAPL...")
        generator = DemoDataGenerator()
        df = generator.generate_stock_data("AAPL", period="5d")
        
        if df.empty:
            print("✗ demo_data: Data generation returned empty DataFrame")
            return False
        
        print(f"✓ demo_data: Successfully generated {len(df)} rows")
        print(f"  Latest price: ${df['Close'].iloc[-1]:.2f}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print("  (Note: This is synthetic data, not real market data)")
        return True
        
    except ImportError:
        print("✗ demo_data: Module not found")
        return False
    except Exception as e:
        print(f"✗ demo_data: Error - {str(e)[:100]}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DATA SOURCE AVAILABILITY TEST")
    print("=" * 60)
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # Test each source
    results['yfinance'] = test_yfinance()
    results['pandas_datareader'] = test_pandas_datareader()
    results['demo_data'] = test_demo_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    available_sources = []
    for source, available in results.items():
        status = "✓ AVAILABLE" if available else "✗ UNAVAILABLE"
        print(f"{source:20s} - {status}")
        if available:
            available_sources.append(source)
    
    print("\n" + "-" * 60)
    if available_sources:
        print(f"Available sources: {', '.join(available_sources)}")
        print("\nRecommendation: Use these sources in order:")
        for i, source in enumerate(available_sources, 1):
            print(f"  {i}. {source}")
    else:
        print("⚠️  WARNING: No data sources are available!")
        print("   Please check your internet connection and dependencies.")
    
    print("=" * 60)
    
    return 0 if available_sources else 1

if __name__ == "__main__":
    sys.exit(main())

