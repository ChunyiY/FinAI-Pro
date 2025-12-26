"""
Demo Data Generator Module

Generates synthetic stock data for demonstration purposes when API is unavailable.
"""
from typing import Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DemoDataGenerator:
    """Generates realistic synthetic stock data for demos."""
    
    def __init__(self, seed: int = 42) -> None:
        """
        Initialize demo data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Predefined stock data templates
        self.stock_templates = {
            'AAPL': {'base_price': 175.0, 'volatility': 0.02, 'trend': 0.0001},
            'MSFT': {'base_price': 380.0, 'volatility': 0.018, 'trend': 0.00008},
            'GOOGL': {'base_price': 140.0, 'volatility': 0.022, 'trend': 0.00012},
            'AMZN': {'base_price': 150.0, 'volatility': 0.025, 'trend': 0.0001},
            'TSLA': {'base_price': 250.0, 'volatility': 0.035, 'trend': 0.00015},
            'META': {'base_price': 320.0, 'volatility': 0.028, 'trend': 0.0001},
            'NVDA': {'base_price': 450.0, 'volatility': 0.03, 'trend': 0.0002},
        }
    
    def generate_stock_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Generate synthetic stock data.
        
        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval
        
        Returns:
            DataFrame with OHLCV data
        """
        # Parse period to get number of days
        days_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        num_days = days_map.get(period, 365)
        
        # Get template or use default
        template = self.stock_templates.get(symbol.upper(), {
            'base_price': 100.0,
            'volatility': 0.02,
            'trend': 0.0001
        })
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        # Generate price data using geometric Brownian motion
        n = len(dates)
        dt = 1 / 252  # Daily time step (assuming 252 trading days per year)
        
        # Random walk with drift
        returns = np.random.normal(
            template['trend'], 
            template['volatility'], 
            n
        )
        
        # Generate prices
        prices = template['base_price'] * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate OHLC with realistic relationships
            high_factor = np.random.uniform(1.0, 1.03)
            low_factor = np.random.uniform(0.97, 1.0)
            
            high = close * high_factor
            low = close * low_factor
            open_price = close * np.random.uniform(0.99, 1.01)
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume (higher volume on larger price changes)
            price_change = abs(close - open_price) / open_price
            base_volume = np.random.uniform(1e6, 5e6)
            volume = base_volume * (1 + price_change * 10)
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': int(volume)
            })
        
        df = pd.DataFrame(data, index=dates[:len(data)])
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        return df
    
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
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get synthetic stock information."""
        template = self.stock_templates.get(symbol.upper(), {
            'base_price': 100.0,
            'volatility': 0.02,
            'trend': 0.0001
        })
        
        current_price = template['base_price'] * np.random.uniform(0.9, 1.1)
        
        return {
            'name': f"{symbol} Corporation",
            'sector': 'Technology',
            'industry': 'Software',
            'market_cap': current_price * np.random.uniform(1e9, 1e12),
            'pe_ratio': np.random.uniform(15, 35),
            'dividend_yield': np.random.uniform(0, 0.03),
            '52w_high': current_price * 1.2,
            '52w_low': current_price * 0.8,
        }

