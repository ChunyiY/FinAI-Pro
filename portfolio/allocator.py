"""
Portfolio Allocation Layer

Transforms risk-adjusted alpha signals into portfolio weights.

Industry Principle:
"Model does not 'decide buy/sell' - it provides risk-aware signals to allocation layer."

This is the critical bridge between alpha signals and actual portfolio construction.
"""

from typing import Dict, Optional, Literal
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class PortfolioAllocator:
    """
    Portfolio Allocator
    
    Converts risk-adjusted alpha signals into portfolio weights.
    
    Default implementation (interpretable, robust):
    w_i ∝ risk_adjusted_alpha_i
    
    With constraints:
    - Cross-sectional normalization (∑|w_i| = 1 or ∑w_i = 1)
    - Single stock maximum weight limit
    - Optional: long-only or market-neutral mode
    """
    
    def __init__(
        self,
        mode: Literal['long_only', 'market_neutral'] = 'long_only',
        max_weight: float = 0.15,  # Maximum weight per stock (15%)
        min_weight: float = 0.0,  # Minimum weight per stock
        normalization: Literal['l1', 'l2', 'sum'] = 'l1'  # Weight normalization method
    ) -> None:
        """
        Initialize portfolio allocator.
        
        Args:
            mode: 'long_only' (all weights ≥ 0) or 'market_neutral' (weights sum to 0)
            max_weight: Maximum weight per stock
            min_weight: Minimum weight per stock
            normalization: 'l1' (∑|w_i| = 1), 'l2' (||w||_2 = 1), or 'sum' (∑w_i = 1)
        """
        self.mode = mode
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.normalization = normalization
        
        if mode == 'market_neutral' and normalization != 'sum':
            # Market neutral requires sum normalization
            self.normalization = 'sum'
    
    def allocate(
        self,
        risk_adjusted_alpha_df: pd.DataFrame,
        target_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Allocate portfolio weights from risk-adjusted alpha signals.
        
        Process:
        1. Extract risk_adjusted_alpha for target date
        2. Apply mode constraints (long-only or market-neutral)
        3. Apply min/max weight constraints
        4. Normalize weights
        
        Args:
            risk_adjusted_alpha_df: DataFrame with columns [date, ticker, risk_adjusted_alpha, ...]
            target_date: Target date (default: most recent)
            
        Returns:
            DataFrame with columns [date, ticker, risk_adjusted_alpha, weight]
        """
        if 'risk_adjusted_alpha' not in risk_adjusted_alpha_df.columns:
            raise ValueError("risk_adjusted_alpha_df must contain 'risk_adjusted_alpha' column")
        
        if 'date' not in risk_adjusted_alpha_df.columns:
            raise ValueError("risk_adjusted_alpha_df must contain 'date' column")
        
        result_df = risk_adjusted_alpha_df.copy()
        
        # Get target date
        if target_date is None:
            target_date = result_df['date'].max()
        
        # Filter to target date
        date_mask = result_df['date'] == target_date
        target_data = result_df[date_mask].copy()
        
        if len(target_data) == 0:
            raise ValueError(f"No data for target date: {target_date}")
        
        # Extract risk-adjusted alpha
        alpha = target_data['risk_adjusted_alpha'].values
        
        # Step 1: Apply mode constraints
        if self.mode == 'long_only':
            # Long-only: set negative alphas to 0
            alpha = np.maximum(alpha, 0)
        elif self.mode == 'market_neutral':
            # Market-neutral: allow negative weights
            # We'll normalize so sum = 0
            pass
        
        # Step 2: Apply min/max weight constraints (before normalization)
        # First, we'll scale alpha to [0, 1] range, then apply constraints after normalization
        
        # Step 3: Calculate raw weights (proportional to alpha)
        if self.normalization == 'l1':
            # L1 normalization: ∑|w_i| = 1
            raw_weights = alpha
            # Handle market-neutral case
            if self.mode == 'market_neutral':
                # For market-neutral, we want sum = 0, so we'll split into long/short
                positive_alpha = np.maximum(alpha, 0)
                negative_alpha = np.minimum(alpha, 0)
                
                # Normalize positive and negative separately
                pos_sum = np.sum(np.abs(positive_alpha))
                neg_sum = np.sum(np.abs(negative_alpha))
                
                if pos_sum > 0:
                    positive_weights = positive_alpha / pos_sum * 0.5
                else:
                    positive_weights = np.zeros_like(positive_alpha)
                
                if neg_sum > 0:
                    negative_weights = negative_alpha / neg_sum * 0.5
                else:
                    negative_weights = np.zeros_like(negative_alpha)
                
                raw_weights = positive_weights + negative_weights
            else:
                # Long-only: normalize by sum of absolute values
                abs_sum = np.sum(np.abs(raw_weights))
                if abs_sum > 0:
                    raw_weights = raw_weights / abs_sum
                else:
                    raw_weights = np.ones_like(raw_weights) / len(raw_weights)
        
        elif self.normalization == 'l2':
            # L2 normalization: ||w||_2 = 1
            raw_weights = alpha
            l2_norm = np.linalg.norm(raw_weights)
            if l2_norm > 0:
                raw_weights = raw_weights / l2_norm
            else:
                raw_weights = np.ones_like(raw_weights) / np.sqrt(len(raw_weights))
        
        elif self.normalization == 'sum':
            # Sum normalization: ∑w_i = 1 (or 0 for market-neutral)
            raw_weights = alpha
            if self.mode == 'market_neutral':
                # Market-neutral: sum should be 0
                total = np.sum(raw_weights)
                if abs(total) > 1e-8:
                    # Scale to make sum = 0 (not exactly, but close)
                    # Actually, for market-neutral, we want equal long/short
                    positive_alpha = np.maximum(alpha, 0)
                    negative_alpha = np.minimum(alpha, 0)
                    
                    pos_sum = np.sum(positive_alpha)
                    neg_sum = np.sum(np.abs(negative_alpha))
                    
                    if pos_sum > 0 and neg_sum > 0:
                        # Normalize to have equal long/short exposure
                        positive_weights = positive_alpha / pos_sum * 0.5
                        negative_weights = negative_alpha / neg_sum * 0.5
                        raw_weights = positive_weights + negative_weights
                    elif pos_sum > 0:
                        raw_weights = positive_alpha / pos_sum
                    elif neg_sum > 0:
                        raw_weights = negative_alpha / neg_sum
                    else:
                        raw_weights = np.zeros_like(alpha)
                else:
                    raw_weights = raw_weights / np.sum(np.abs(raw_weights)) if np.sum(np.abs(raw_weights)) > 0 else np.zeros_like(raw_weights)
            else:
                # Long-only: sum = 1
                total = np.sum(raw_weights)
                if total > 0:
                    raw_weights = raw_weights / total
                else:
                    raw_weights = np.ones_like(raw_weights) / len(raw_weights)
        
        # Step 4: Apply min/max weight constraints
        raw_weights = np.clip(raw_weights, self.min_weight, self.max_weight)
        
        # Step 5: Renormalize after clipping (to maintain normalization property)
        if self.normalization == 'l1':
            abs_sum = np.sum(np.abs(raw_weights))
            if abs_sum > 0:
                raw_weights = raw_weights / abs_sum
        elif self.normalization == 'l2':
            l2_norm = np.linalg.norm(raw_weights)
            if l2_norm > 0:
                raw_weights = raw_weights / l2_norm
        elif self.normalization == 'sum':
            total = np.sum(raw_weights)
            if abs(total) > 1e-8:
                if self.mode == 'market_neutral':
                    # For market-neutral, we want sum ≈ 0, so we adjust
                    # Split into long/short and renormalize
                    positive_weights = np.maximum(raw_weights, 0)
                    negative_weights = np.minimum(raw_weights, 0)
                    
                    pos_sum = np.sum(positive_weights)
                    neg_sum = np.sum(np.abs(negative_weights))
                    
                    if pos_sum > 0 and neg_sum > 0:
                        positive_weights = positive_weights / pos_sum * 0.5
                        negative_weights = negative_weights / neg_sum * 0.5
                        raw_weights = positive_weights + negative_weights
                    elif pos_sum > 0:
                        raw_weights = positive_weights / pos_sum
                    elif neg_sum > 0:
                        raw_weights = negative_weights / neg_sum
                else:
                    raw_weights = raw_weights / total
        
        # Store weights
        target_data['weight'] = raw_weights
        
        # Update result dataframe
        result_df.loc[date_mask, 'weight'] = target_data['weight'].values
        
        return result_df
    
    def get_allocation_summary(
        self,
        allocation_df: pd.DataFrame,
        target_date: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Get summary statistics for portfolio allocation.
        
        Args:
            allocation_df: DataFrame with weight column
            target_date: Target date (default: most recent)
            
        Returns:
            Dictionary with allocation statistics
        """
        if target_date is None:
            target_date = allocation_df['date'].max()
        
        date_mask = allocation_df['date'] == target_date
        target_data = allocation_df[date_mask].copy()
        
        if 'weight' not in target_data.columns:
            raise ValueError("allocation_df must contain 'weight' column")
        
        weights = target_data['weight'].values
        
        summary = {
            'total_stocks': len(target_data),
            'long_positions': np.sum(weights > 0),
            'short_positions': np.sum(weights < 0),
            'zero_positions': np.sum(weights == 0),
            'total_weight': np.sum(weights),
            'total_abs_weight': np.sum(np.abs(weights)),
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'concentration': np.sum(weights ** 2),  # Herfindahl index
            'top_5_weight': np.sum(np.sort(np.abs(weights))[-5:]) if len(weights) >= 5 else np.sum(np.abs(weights))
        }
        
        return summary

