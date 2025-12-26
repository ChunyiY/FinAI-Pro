"""
Real Data Fetcher - Alpha Vantage Only

Fetches real stock market data exclusively from Alpha Vantage API.
"""
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import os
import time

warnings.filterwarnings('ignore')

# Try to load API key from config
try:
    from config import ALPHA_VANTAGE_API_KEY as CONFIG_ALPHA_KEY
    if CONFIG_ALPHA_KEY and CONFIG_ALPHA_KEY != 'demo':
        os.environ['ALPHA_VANTAGE_API_KEY'] = CONFIG_ALPHA_KEY
except ImportError:
    pass


class RealDataFetcher:
    """Fetches real stock market data from Alpha Vantage API only."""
    
    def __init__(self) -> None:
        """Initialize Alpha Vantage data fetcher."""
        self.cache: Dict[str, pd.DataFrame] = {}
        self.api_key = None
        
        # Get API key
        try:
            from config import ALPHA_VANTAGE_API_KEY as CONFIG_KEY
            if CONFIG_KEY and CONFIG_KEY != 'demo' and CONFIG_KEY != '':
                self.api_key = CONFIG_KEY
        except ImportError:
            pass
        
        if not self.api_key:
            self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if not self.api_key or self.api_key == 'demo' or self.api_key == '':
            raise Exception(
                "Alpha Vantage API key not found! "
                "Please set ALPHA_VANTAGE_API_KEY in config.py or as environment variable."
            )
        
        print(f"✓ Alpha Vantage initialized with API key: {self.api_key[:10]}...")
    
    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch real stock data from Alpha Vantage.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y')
            interval: Data interval ('1d', '1wk', '1mo')
        
        Returns:
            DataFrame with real stock data
        
        Raises:
            Exception: If data cannot be fetched
        """
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # Rate limit: 5 calls per minute for free tier
        time.sleep(12)  # Wait 12 seconds between calls to stay under limit
        
        try:
            df = self._fetch_alpha_vantage(symbol, period, interval)
            if df is not None and not df.empty:
                # Add technical indicators
                df = self._add_technical_indicators(df)
                self.cache[cache_key] = df.copy()
                return df
            else:
                raise Exception(f"Empty data returned for {symbol}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'note' in error_msg or 'api call' in error_msg:
                raise Exception(
                    f"Alpha Vantage rate limit reached. "
                    f"Free tier allows 5 calls per minute. "
                    f"Please wait a moment and try again."
                )
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def _fetch_alpha_vantage(
        self, 
        symbol: str, 
        period: str, 
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage TIME_SERIES_DAILY endpoint."""
        import requests
        
        # Map interval to Alpha Vantage function
        # For daily data, use TIME_SERIES_DAILY
        function = 'TIME_SERIES_DAILY'
        
        # Try 'full' first (premium feature), fallback to 'compact' if not available
        # 'full' returns complete history (20+ years) - premium feature
        # 'compact' returns last 100 data points - free tier
        outputsize_options = ['full', 'compact']
        
        url = 'https://www.alphavantage.co/query'
        
        # Try 'full' first, then fallback to 'compact' if premium not available
        for outputsize in outputsize_options:
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': outputsize,
                'datatype': 'json'
            }
            
            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    raise Exception(f"Alpha Vantage API Error: {data['Error Message']}")
                
                if 'Note' in data:
                    raise Exception(f"Alpha Vantage API Note: {data['Note']}")
                
                # Check if this is a premium feature error
                if 'Information' in data:
                    info_msg = data['Information'].lower()
                    # If it's a premium feature error and we're trying 'full', fallback to 'compact'
                    if 'premium' in info_msg and outputsize == 'full':
                        print(f"⚠️  Premium feature not available. Using 'compact' mode (last ~100 data points).")
                        continue  # Try 'compact' instead
                    else:
                        raise Exception(f"Alpha Vantage API Info: {data['Information']}")
            
                # Check for time series data
                if 'Time Series (Daily)' not in data:
                    # Try to find what keys are available for debugging
                    available_keys = list(data.keys())
                    raise Exception(
                        f"No time series data found. "
                        f"Response keys: {available_keys}. "
                        f"Full response: {str(data)[:200]}"
                    )
                
                # Convert to DataFrame
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                
                # Rename columns (Alpha Vantage uses '1. open', '2. high', etc.)
                column_mapping = {
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                }
                df = df.rename(columns=column_mapping)
                
                # Convert to float
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Sort by date
                df = df.sort_index()
                
                # Filter by period if needed
                # Note: 'compact' returns ~100 days, 'full' returns all history
                end_date = datetime.now()
                period_map = {
                    '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
                    '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650
                }
                days = period_map.get(period, 365)
                start_date = end_date - timedelta(days=days)
                df = df[df.index >= start_date]
                
                if df.empty:
                    return None
                
                # Successfully got data, return it
                if outputsize == 'full':
                    print(f"✓ Successfully fetched full historical data for {symbol}")
                else:
                    print(f"✓ Fetched data for {symbol} (compact mode, ~100 days)")
                
                return df
                
            except requests.exceptions.RequestException as e:
                # Network error, try next option or raise
                if outputsize == 'full':
                    print(f"⚠️  Network error with 'full' mode, trying 'compact'...")
                    continue
                raise Exception(f"Network error: {str(e)}")
            except Exception as e:
                # Other errors, if we're trying 'full', fallback to 'compact'
                if outputsize == 'full' and 'premium' not in str(e).lower():
                    print(f"⚠️  Error with 'full' mode: {str(e)[:100]}. Trying 'compact'...")
                    continue
                # If we're already on 'compact' or it's not a premium error, raise
                raise Exception(f"Error processing Alpha Vantage response: {str(e)}")
        
        # If we get here, both 'full' and 'compact' failed
        raise Exception("Failed to fetch data with both 'full' and 'compact' outputsize options")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume moving average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get stock information using Alpha Vantage OVERVIEW endpoint."""
        import requests
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            # Rate limit
            time.sleep(12)
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data or 'Note' in data:
                return {
                    'name': symbol,
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'market_cap': 0,
                    'pe_ratio': 0,
                    'dividend_yield': 0,
                    '52w_high': 0,
                    '52w_low': 0,
                }
            
            return {
                'name': data.get('Name', symbol),
                'sector': data.get('Sector', 'N/A'),
                'industry': data.get('Industry', 'N/A'),
                'market_cap': float(data.get('MarketCapitalization', 0) or 0),
                'pe_ratio': float(data.get('PERatio', 0) or 0),
                'dividend_yield': float(data.get('DividendYield', 0) or 0),
                '52w_high': float(data.get('52WeekHigh', 0) or 0),
                '52w_low': float(data.get('52WeekLow', 0) or 0),
            }
        except Exception:
            return {
                'name': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'pe_ratio': 0,
                'dividend_yield': 0,
                '52w_high': 0,
                '52w_low': 0,
            }
    
    def get_multiple_stocks(
        self, 
        symbols: list, 
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks."""
        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                data[symbol] = self.get_stock_data(symbol, period)
            except Exception as e:
                print(f"Warning: Unable to fetch data for {symbol}: {e}")
        return data
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources (only Alpha Vantage)."""
        return ['alpha_vantage']
