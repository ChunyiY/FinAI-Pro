"""
Multi-Source Stock Data Fetcher

Supports multiple data sources with automatic failover to avoid API issues.
"""
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')


class MultiSourceDataFetcher:
    """Fetches stock data from multiple sources with automatic failover."""
    
    def __init__(
        self, 
        preferred_source: str = "yfinance",
        use_demo_on_error: bool = True
    ) -> None:
        """
        Initialize multi-source data fetcher.
        
        Args:
            preferred_source: Preferred data source ('yfinance', 'pandas_datareader', 'demo')
            use_demo_on_error: Use demo data if all sources fail
        """
        self.preferred_source = preferred_source
        self.use_demo_on_error = use_demo_on_error
        self.cache: Dict[str, pd.DataFrame] = {}
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
        Fetch stock data trying multiple sources.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period
            interval: Data interval
        
        Returns:
            DataFrame with stock data
        """
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # Try sources in order
        sources = self._get_source_order()
        
        for source in sources:
            try:
                df = self._fetch_from_source(source, symbol, period, interval)
                if df is not None and not df.empty:
                    # Add technical indicators
                    df = self._add_technical_indicators(df)
                    self.cache[cache_key] = df.copy()
                    return df
            except Exception as e:
                print(f"⚠️  {source} failed: {str(e)[:100]}")
                continue
        
        # All sources failed, try demo data
        if self.use_demo_on_error and self._demo_generator:
            print(f"⚠️  All APIs unavailable. Using demo data for {symbol}...")
            try:
                demo_data = self._demo_generator.generate_stock_data(symbol, period, interval)
                self.cache[cache_key] = demo_data.copy()
                return demo_data
            except Exception as e:
                print(f"Demo data generation failed: {e}")
        
        raise Exception(f"Unable to fetch data for {symbol} from any source")
    
    def _get_source_order(self) -> List[str]:
        """Get ordered list of data sources to try."""
        sources = []
        
        # Add preferred source first
        if self.preferred_source != "demo":
            sources.append(self.preferred_source)
        
        # Add other sources
        if self.preferred_source != "yfinance":
            sources.append("yfinance")
        
        if self.preferred_source != "pandas_datareader":
            sources.append("pandas_datareader")
        
        return sources
    
    def _fetch_from_source(
        self, 
        source: str, 
        symbol: str, 
        period: str, 
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from a specific source."""
        if source == "yfinance":
            return self._fetch_yfinance(symbol, period, interval)
        elif source == "pandas_datareader":
            return self._fetch_pandas_datareader(symbol, period)
        else:
            return None
    
    def _fetch_yfinance(
        self, 
        symbol: str, 
        period: str, 
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df if not df.empty else None
        except Exception as e:
            error_msg = str(e).lower()
            # Skip rate limit errors quickly
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                raise Exception("Rate limited")
            raise
    
    def _fetch_pandas_datareader(
        self, 
        symbol: str, 
        period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data using pandas_datareader (Yahoo Finance backend)."""
        try:
            import pandas_datareader.data as web
            from datetime import datetime, timedelta
            
            # Convert period to dates
            end_date = datetime.now()
            period_map = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
                '6mo': 180, '1y': 365, '2y': 730
            }
            days = period_map.get(period, 365)
            start_date = end_date - timedelta(days=days)
            
            df = web.get_data_yahoo(symbol, start=start_date, end=end_date)
            return df if not df.empty else None
        except ImportError:
            # pandas_datareader not installed
            return None
        except Exception:
            # API failed
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
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
        """Get stock information."""
        # Try yfinance first
        try:
            import yfinance as yf
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
        except Exception:
            pass
        
        # Fallback to demo data
        if self.use_demo_on_error and self._demo_generator:
            return self._demo_generator.get_stock_info(symbol)
        
        return {'error': 'Unable to fetch stock info'}
    
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

