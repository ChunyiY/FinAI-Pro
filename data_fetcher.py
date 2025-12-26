"""
Stock Data Fetcher Module

Fetches stock data from yfinance API and calculates technical indicators.
Falls back to demo data generator when API is unavailable.
"""
from typing import Dict, Optional
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')


class StockDataFetcher:
    """Fetches and processes stock market data."""
    
    def __init__(
        self, 
        max_retries: int = 3, 
        retry_delay: float = 2.0,
        use_demo_on_error: bool = True
    ) -> None:
        """
        Initialize the data fetcher.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay in seconds between retries
            use_demo_on_error: Use demo data if API fails
        """
        self.cache: Dict[str, pd.DataFrame] = {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_demo_on_error = use_demo_on_error
        self._demo_generator = None
        
        if use_demo_on_error:
            try:
                from demo_data import DemoDataGenerator
                self._demo_generator = DemoDataGenerator()
            except ImportError:
                pass
    
    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data with retry mechanism.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame containing OHLCV data with technical indicators
        
        Raises:
            ValueError: If data cannot be fetched or is empty
            Exception: For network/timeout errors after retries
        """
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    raise ValueError(f"Unable to fetch data for {symbol}. Please check the symbol.")
                
                # Add technical indicators
                df = self._add_technical_indicators(df)
                
                self.cache[cache_key] = df.copy()
                return df
                
            except ValueError:
                raise
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check for rate limiting (most important to handle)
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    if attempt < self.max_retries - 1:
                        # Use longer delay for rate limiting (exponential backoff)
                        wait_time = self.retry_delay * (2 ** (attempt + 1))  # 4s, 8s, 16s
                        print(f"Rate limit detected (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(
                            f"Rate limit exceeded: Yahoo Finance API is rate limiting requests. "
                            f"Please wait a few minutes before trying again. "
                            f"Error: {str(e)}"
                        )
                # Check for timeout or connection errors
                elif 'timeout' in error_msg or 'connection' in error_msg or 'curl' in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (attempt + 1)
                        print(f"Network timeout/error (attempt {attempt + 1}/{self.max_retries}). Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(
                            f"Network timeout: Unable to fetch data for {symbol} after {self.max_retries} attempts. "
                            f"Please check your internet connection or try again later. "
                            f"Error: {str(e)}"
                        )
                else:
                    # For other errors, raise immediately
                    raise Exception(f"Error fetching data for {symbol}: {str(e)}")
        
        # If we get here, all retries failed
        # Try demo data as fallback
        if self.use_demo_on_error and self._demo_generator:
            print(f"⚠️  API unavailable. Using demo data for {symbol}...")
            try:
                demo_data = self._demo_generator.generate_stock_data(symbol, period, interval)
                self.cache[cache_key] = demo_data.copy()
                return demo_data
            except Exception as demo_error:
                print(f"Demo data generation also failed: {demo_error}")
        
        raise Exception(
            f"Failed to fetch data for {symbol} after {self.max_retries} attempts. "
            f"Last error: {str(last_error)}"
        )
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added technical indicators
        """
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index)
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
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: Series of closing prices
            period: Period for RSI calculation (default: 14)
        
        Returns:
            Series containing RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get basic stock information with retry mechanism.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary containing stock information
        """
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return {
                    'name': info.get('longName', symbol),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    '52w_high': info.get('fiftyTwoWeekHigh', 0),
                    '52w_low': info.get('fiftyTwoWeekLow', 0),
                }
            except Exception as e:
                error_msg = str(e).lower()
                # Handle rate limiting with longer delays
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** (attempt + 1))
                        time.sleep(wait_time)
                        continue
                elif ('timeout' in error_msg or 'connection' in error_msg) and attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        # Fallback to demo data if available
        if self.use_demo_on_error and self._demo_generator:
            print(f"⚠️  Using demo stock info for {symbol}...")
            return self._demo_generator.get_stock_info(symbol)
        
        return {'error': 'Failed to fetch stock info after retries'}
    
    def get_multiple_stocks(
        self, 
        symbols: list, 
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period
        
        Returns:
            Dictionary mapping symbols to their dataframes
        """
        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                data[symbol] = self.get_stock_data(symbol, period)
            except Exception as e:
                print(f"Warning: Unable to fetch data for {symbol}: {e}")
        return data
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current stock price with retry mechanism.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Current price as float
        
        Raises:
            Exception: If price cannot be fetched after retries
        """
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    return float(data['Close'].iloc[-1])
                else:
                    # Fallback to daily data if minute data unavailable
                    data = ticker.history(period="5d")
                    return float(data['Close'].iloc[-1])
            except Exception as e:
                error_msg = str(e).lower()
                # Handle rate limiting with longer delays
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** (attempt + 1))
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(
                            f"Rate limit exceeded: Unable to fetch current price for {symbol}. "
                            f"Please wait a few minutes before trying again."
                        )
                elif ('timeout' in error_msg or 'connection' in error_msg) and attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise Exception(f"Error fetching current price for {symbol}: {str(e)}")
        
        raise Exception(f"Failed to fetch current price for {symbol} after {self.max_retries} attempts")
