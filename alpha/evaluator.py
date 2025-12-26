"""
Cross-Sectional Alpha Evaluation Module

Industry-standard evaluation metrics for cross-sectional alpha signals.
Focus on Rank IC, Top-K vs Bottom-K returns, not ML accuracy metrics.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')


class CrossSectionalEvaluator:
    """
    Evaluator for cross-sectional alpha signals.
    
    Industry Perspective:
    - Not ML accuracy (Accuracy, F1, confusion matrix)
    - Focus on Rank IC, Top-K vs Bottom-K performance
    - Alpha distribution stability
    - Turnover proxy
    """
    
    def __init__(self, top_k_ratio: float = 0.2) -> None:
        """
        Initialize evaluator.
        
        Args:
            top_k_ratio: Ratio for Top-K / Bottom-K analysis (default: 0.2 = top 20%)
        """
        self.top_k_ratio = top_k_ratio
    
    def calculate_rank_ic(
        self,
        predicted_alpha: np.ndarray,
        actual_excess_return: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Rank IC (Information Coefficient).
        
        Rank IC = Spearman correlation between predicted alpha rank and actual excess return rank.
        This is the industry standard for evaluating cross-sectional alpha signals.
        
        Args:
            predicted_alpha: Predicted alpha scores
            actual_excess_return: Actual cross-sectional excess returns
            
        Returns:
            Dictionary with rank_ic, p_value, and significance
        """
        if len(predicted_alpha) != len(actual_excess_return):
            raise ValueError("predicted_alpha and actual_excess_return must have same length")
        
        if len(predicted_alpha) < 2:
            return {
                'rank_ic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        
        # Calculate Spearman correlation
        rank_ic, p_value = spearmanr(predicted_alpha, actual_excess_return)
        
        # Handle NaN cases
        if np.isnan(rank_ic):
            rank_ic = 0.0
        if np.isnan(p_value):
            p_value = 1.0
        
        return {
            'rank_ic': rank_ic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def calculate_mean_rank_ic(
        self,
        predictions_by_date: Dict[pd.Timestamp, np.ndarray],
        actual_by_date: Dict[pd.Timestamp, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate mean Rank IC across multiple dates.
        
        Args:
            predictions_by_date: Dictionary mapping date to predicted alpha array
            actual_by_date: Dictionary mapping date to actual excess return array
            
        Returns:
            Dictionary with mean_rank_ic, std_rank_ic, and hit_rate
        """
        rank_ics = []
        significant_count = 0
        
        for date in predictions_by_date.keys():
            if date not in actual_by_date:
                continue
            
            pred = predictions_by_date[date]
            actual = actual_by_date[date]
            
            if len(pred) != len(actual) or len(pred) < 2:
                continue
            
            result = self.calculate_rank_ic(pred, actual)
            rank_ics.append(result['rank_ic'])
            
            if result['significant']:
                significant_count += 1
        
        if len(rank_ics) == 0:
            return {
                'mean_rank_ic': 0.0,
                'std_rank_ic': 0.0,
                'hit_rate': 0.0,
                'n_dates': 0
            }
        
        rank_ics = np.array(rank_ics)
        
        return {
            'mean_rank_ic': np.mean(rank_ics),
            'std_rank_ic': np.std(rank_ics),
            'hit_rate': significant_count / len(rank_ics),
            'n_dates': len(rank_ics)
        }
    
    def calculate_topk_vs_bottomk(
        self,
        predicted_alpha: np.ndarray,
        actual_excess_return: np.ndarray,
        tickers: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate Top-K vs Bottom-K average return.
        
        Compare average actual return of top K% stocks (by predicted alpha)
        vs bottom K% stocks.
        
        Args:
            predicted_alpha: Predicted alpha scores
            actual_excess_return: Actual excess returns
            tickers: Optional ticker array for reporting
            
        Returns:
            Dictionary with top_k_return, bottom_k_return, spread, and other metrics
        """
        if len(predicted_alpha) != len(actual_excess_return):
            raise ValueError("predicted_alpha and actual_excess_return must have same length")
        
        n = len(predicted_alpha)
        k = max(1, int(n * self.top_k_ratio))
        
        if n < 2 * k:
            return {
                'top_k_return': 0.0,
                'bottom_k_return': 0.0,
                'spread': 0.0,
                'top_k_count': 0,
                'bottom_k_count': 0
            }
        
        # Rank by predicted alpha
        top_k_indices = np.argsort(predicted_alpha)[-k:]
        bottom_k_indices = np.argsort(predicted_alpha)[:k]
        
        # Calculate average returns
        top_k_return = np.mean(actual_excess_return[top_k_indices])
        bottom_k_return = np.mean(actual_excess_return[bottom_k_indices])
        spread = top_k_return - bottom_k_return
        
        result = {
            'top_k_return': top_k_return,
            'bottom_k_return': bottom_k_return,
            'spread': spread,
            'top_k_count': k,
            'bottom_k_count': k
        }
        
        if tickers is not None:
            result['top_k_tickers'] = tickers[top_k_indices].tolist()
            result['bottom_k_tickers'] = tickers[bottom_k_indices].tolist()
        
        return result
    
    def evaluate_alpha_stability(
        self,
        alpha_by_date: Dict[pd.Timestamp, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate alpha distribution stability across time.
        
        Measures how stable alpha signals are over time (lower turnover = more stable).
        
        Args:
            alpha_by_date: Dictionary mapping date to alpha array
            
        Returns:
            Dictionary with stability metrics
        """
        if len(alpha_by_date) < 2:
            return {
                'mean_std': 0.0,
                'mean_range': 0.0,
                'stability_score': 0.0
            }
        
        dates = sorted(alpha_by_date.keys())
        alphas = [alpha_by_date[date] for date in dates]
        
        # Calculate cross-sectional std for each date
        cross_sectional_stds = [np.std(alpha) for alpha in alphas]
        cross_sectional_ranges = [np.max(alpha) - np.min(alpha) for alpha in alphas]
        
        # Stability: lower variation in cross-sectional distribution = more stable
        mean_std = np.mean(cross_sectional_stds)
        mean_range = np.mean(cross_sectional_ranges)
        
        # Stability score (inverse of coefficient of variation)
        if mean_std > 0:
            stability_score = 1.0 / (1.0 + np.std(cross_sectional_stds) / mean_std)
        else:
            stability_score = 0.0
        
        return {
            'mean_std': mean_std,
            'mean_range': mean_range,
            'stability_score': stability_score
        }
    
    def calculate_turnover_proxy(
        self,
        weights_by_date: Dict[pd.Timestamp, np.ndarray],
        tickers: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate turnover proxy (simplified).
        
        Measures how much portfolio weights change between dates.
        Higher turnover = more trading required.
        
        Args:
            weights_by_date: Dictionary mapping date to weight array
            tickers: Optional ticker array (must be consistent across dates)
            
        Returns:
            Dictionary with turnover metrics
        """
        if len(weights_by_date) < 2:
            return {
                'mean_turnover': 0.0,
                'max_turnover': 0.0
            }
        
        dates = sorted(weights_by_date.keys())
        turnovers = []
        
        for i in range(1, len(dates)):
            prev_weights = weights_by_date[dates[i-1]]
            curr_weights = weights_by_date[dates[i]]
            
            if len(prev_weights) != len(curr_weights):
                continue
            
            # Turnover = sum of absolute changes in weights
            turnover = np.sum(np.abs(curr_weights - prev_weights))
            turnovers.append(turnover)
        
        if len(turnovers) == 0:
            return {
                'mean_turnover': 0.0,
                'max_turnover': 0.0
            }
        
        return {
            'mean_turnover': np.mean(turnovers),
            'max_turnover': np.max(turnovers),
            'std_turnover': np.std(turnovers)
        }
    
    def comprehensive_evaluation(
        self,
        predictions_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        weights_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Comprehensive evaluation of cross-sectional alpha engine.
        
        Args:
            predictions_df: DataFrame with [date, ticker, alpha_raw, ...]
            actual_df: DataFrame with [date, ticker, actual_excess_return]
            weights_df: Optional DataFrame with [date, ticker, weight]
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Group by date
        predictions_by_date = {}
        actual_by_date = {}
        tickers_by_date = {}
        
        for date in predictions_df['date'].unique():
            date_mask = predictions_df['date'] == date
            pred_data = predictions_df[date_mask]
            actual_data = actual_df[actual_df['date'] == date]
            
            if len(pred_data) == 0 or len(actual_data) == 0:
                continue
            
            # Align by ticker
            pred_dict = dict(zip(pred_data['ticker'], pred_data['alpha_raw']))
            actual_dict = dict(zip(actual_data['ticker'], actual_data.get('actual_excess_return', [])))
            
            common_tickers = set(pred_dict.keys()) & set(actual_dict.keys())
            
            if len(common_tickers) < 2:
                continue
            
            common_tickers = sorted(common_tickers)
            predictions_by_date[date] = np.array([pred_dict[t] for t in common_tickers])
            actual_by_date[date] = np.array([actual_dict[t] for t in common_tickers])
            tickers_by_date[date] = np.array(common_tickers)
        
        # Mean Rank IC
        mean_rank_ic_result = self.calculate_mean_rank_ic(predictions_by_date, actual_by_date)
        results['mean_rank_ic'] = mean_rank_ic_result
        
        # Top-K vs Bottom-K (aggregate across dates)
        all_topk_results = []
        for date in predictions_by_date.keys():
            topk_result = self.calculate_topk_vs_bottomk(
                predictions_by_date[date],
                actual_by_date[date],
                tickers_by_date[date]
            )
            all_topk_results.append(topk_result)
        
        if all_topk_results:
            results['topk_analysis'] = {
                'mean_top_k_return': np.mean([r['top_k_return'] for r in all_topk_results]),
                'mean_bottom_k_return': np.mean([r['bottom_k_return'] for r in all_topk_results]),
                'mean_spread': np.mean([r['spread'] for r in all_topk_results])
            }
        
        # Alpha stability
        alpha_by_date = {date: predictions_by_date[date] for date in predictions_by_date.keys()}
        stability_result = self.evaluate_alpha_stability(alpha_by_date)
        results['alpha_stability'] = stability_result
        
        # Turnover (if weights provided)
        if weights_df is not None:
            weights_by_date = {}
            for date in weights_df['date'].unique():
                date_mask = weights_df['date'] == date
                weight_data = weights_df[date_mask]
                weights_by_date[date] = weight_data['weight'].values
            
            turnover_result = self.calculate_turnover_proxy(weights_by_date)
            results['turnover'] = turnover_result
        
        return results

