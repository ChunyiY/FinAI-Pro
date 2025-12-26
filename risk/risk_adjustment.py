"""
Risk Adjustment Layer

Critical component: Any unadjusted alpha is NOT suitable for portfolio allocation.
This layer transforms noisy raw alpha signals into risk-adjusted signals.

Key Operations:
1. Cross-sectional standardization (z-score normalization)
2. Volatility penalty (alpha / volatility)
3. Optional: Uncertainty estimation and stability penalties
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class RiskAdjustmentLayer:
    """
    Risk Adjustment Layer for Alpha Signals
    
    Industry Principle:
    "Any unadjusted alpha is NOT suitable for portfolio allocation."
    
    This layer transforms raw alpha signals (alpha_raw) into risk-adjusted
    alpha signals (risk_adjusted_alpha) that can be used for portfolio construction.
    
    Process:
    1. Cross-sectional z-score normalization (within same date)
    2. Volatility penalty (divide by stock volatility)
    3. Optional: Uncertainty/stability adjustments
    """
    
    def __init__(
        self,
        volatility_window: int = 20,  # Rolling window for volatility calculation
        min_volatility: float = 0.01,  # Minimum volatility floor (1% daily)
        use_uncertainty_penalty: bool = False,  # Optional: uncertainty adjustment
        uncertainty_window: int = 30  # Window for uncertainty estimation
    ) -> None:
        """
        Initialize risk adjustment layer.
        
        Args:
            volatility_window: Rolling window for volatility calculation
            min_volatility: Minimum volatility floor to avoid division by zero
            use_uncertainty_penalty: Whether to apply uncertainty penalty
            uncertainty_window: Window for rolling prediction error estimation
        """
        self.volatility_window = volatility_window
        self.min_volatility = min_volatility
        self.use_uncertainty_penalty = use_uncertainty_penalty
        self.uncertainty_window = uncertainty_window
    
    def calculate_volatility(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        target_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Calculate rolling volatility for each stock.
        
        Args:
            stocks_data: Dictionary mapping ticker to DataFrame
            target_date: Target date (default: most recent)
            
        Returns:
            Series mapping ticker to volatility
        """
        volatilities = {}
        
        for ticker, data in stocks_data.items():
            if data.empty or 'Close' not in data.columns:
                continue
            
            # Get date index
            if isinstance(data.index, pd.DatetimeIndex):
                dates = data.index
            elif 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
            else:
                continue
            
            # Select target date
            if target_date is None:
                target_date = dates.max()
            
            # Get data up to target date
            mask = dates <= target_date
            price_data = data.loc[mask, 'Close'] if hasattr(data.loc, '__call__') else data[mask]['Close']
            
            if len(price_data) < self.volatility_window:
                # Use available data if insufficient
                window = min(len(price_data), self.volatility_window)
            else:
                window = self.volatility_window
            
            if window < 5:
                volatilities[ticker] = self.min_volatility
                continue
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            if len(returns) < window:
                volatilities[ticker] = self.min_volatility
                continue
            
            # Rolling volatility (annualized, then converted to daily)
            volatility = returns.tail(window).std() * np.sqrt(252) / np.sqrt(252)  # Daily volatility
            volatility = max(volatility, self.min_volatility)
            
            volatilities[ticker] = volatility
        
        return pd.Series(volatilities)
    
    def adjust_alpha(
        self,
        alpha_raw_df: pd.DataFrame,
        stocks_data: Dict[str, pd.DataFrame],
        target_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Apply risk adjustment to raw alpha signals.
        
        Process:
        1. Cross-sectional z-score normalization (alpha_z)
        2. Volatility penalty (risk_adjusted_alpha = alpha_z / volatility)
        3. Optional: Uncertainty penalty
        
        Args:
            alpha_raw_df: DataFrame with columns [date, ticker, alpha_raw]
            stocks_data: Dictionary mapping ticker to DataFrame (for volatility calculation)
            target_date: Target date (default: most recent in alpha_raw_df)
            
        Returns:
            DataFrame with columns [date, ticker, alpha_raw, alpha_z, volatility, risk_adjusted_alpha]
        """
        if 'alpha_raw' not in alpha_raw_df.columns:
            raise ValueError("alpha_raw_df must contain 'alpha_raw' column")
        
        if 'date' not in alpha_raw_df.columns:
            raise ValueError("alpha_raw_df must contain 'date' column")
        
        result_df = alpha_raw_df.copy()
        
        # Get target date
        if target_date is None:
            target_date = result_df['date'].max()
        
        # Filter to target date
        date_mask = result_df['date'] == target_date
        target_alpha = result_df[date_mask].copy()
        
        if len(target_alpha) == 0:
            raise ValueError(f"No data for target date: {target_date}")
        
        # Step 1: Cross-sectional z-score normalization
        # alpha_z_i = zscore(alpha_raw_i across stocks on same date)
        alpha_raw_values = target_alpha['alpha_raw'].values
        alpha_mean = np.mean(alpha_raw_values)
        alpha_std = np.std(alpha_raw_values)
        
        if alpha_std < 1e-8:
            # If all alphas are the same, set z-scores to 0
            alpha_z = np.zeros_like(alpha_raw_values)
        else:
            alpha_z = (alpha_raw_values - alpha_mean) / alpha_std
        
        target_alpha['alpha_z'] = alpha_z
        
        # Step 2: Volatility penalty
        # risk_adjusted_alpha_i = alpha_z_i / volatility_i
        volatilities = self.calculate_volatility(stocks_data, target_date=target_date)
        
        risk_adjusted_alpha = []
        for ticker in target_alpha['ticker']:
            if ticker in volatilities.index:
                vol = volatilities[ticker]
            else:
                vol = self.min_volatility
            
            # Get corresponding alpha_z
            ticker_mask = target_alpha['ticker'] == ticker
            if ticker_mask.any():
                alpha_z_val = target_alpha[ticker_mask]['alpha_z'].iloc[0]
                risk_adj = alpha_z_val / (vol + 1e-8)  # Add small epsilon for numerical stability
            else:
                risk_adj = 0.0
            
            risk_adjusted_alpha.append(risk_adj)
        
        target_alpha['volatility'] = target_alpha['ticker'].map(volatilities).fillna(self.min_volatility)
        target_alpha['risk_adjusted_alpha'] = risk_adjusted_alpha
        
        # Step 3: Optional uncertainty penalty
        if self.use_uncertainty_penalty:
            # Estimate uncertainty from rolling prediction error
            # This is a simplified version - in practice, you'd use prediction intervals
            uncertainty_penalty = np.ones(len(target_alpha))
            target_alpha['uncertainty_penalty'] = uncertainty_penalty
            target_alpha['risk_adjusted_alpha'] = target_alpha['risk_adjusted_alpha'] * uncertainty_penalty
        
        # Update result dataframe
        result_df.loc[date_mask, 'alpha_z'] = target_alpha['alpha_z'].values
        result_df.loc[date_mask, 'volatility'] = target_alpha['volatility'].values
        result_df.loc[date_mask, 'risk_adjusted_alpha'] = target_alpha['risk_adjusted_alpha'].values
        
        return result_df

