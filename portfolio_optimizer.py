"""
Portfolio Optimization Module

Implements Modern Portfolio Theory for portfolio optimization.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """Portfolio optimizer based on Modern Portfolio Theory."""
    
    def __init__(self) -> None:
        """Initialize portfolio optimizer."""
        pass
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from prices.
        
        Args:
            prices: DataFrame with stock prices
        
        Returns:
            DataFrame with returns
        """
        return prices.pct_change().dropna()
    
    def calculate_portfolio_stats(
        self, 
        weights: np.ndarray, 
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio statistics.
        
        Args:
            weights: Array of portfolio weights
            returns: DataFrame with returns
        
        Returns:
            Dictionary with return, volatility, and Sharpe ratio
        """
        # Portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
        
        # Portfolio volatility (standard deviation)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def negative_sharpe(
        self, 
        weights: np.ndarray, 
        returns: pd.DataFrame
    ) -> float:
        """
        Negative Sharpe ratio (for minimization).
        
        Args:
            weights: Portfolio weights
            returns: Returns DataFrame
        
        Returns:
            Negative Sharpe ratio
        """
        stats = self.calculate_portfolio_stats(weights, returns)
        return -stats['sharpe_ratio']
    
    def optimize_portfolio(
        self, 
        returns: pd.DataFrame, 
        risk_free_rate: float = 0.02, 
        method: str = 'max_sharpe'
    ) -> Dict:
        """
        Optimize portfolio.
        
        Args:
            returns: Returns DataFrame
            risk_free_rate: Risk-free rate
            method: Optimization method ('max_sharpe', 'min_volatility', 'max_return')
        
        Returns:
            Dictionary with optimal weights and statistics
        
        Raises:
            ValueError: If unknown optimization method
        """
        num_assets = len(returns.columns)
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial weights (equal weight)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        if method == 'max_sharpe':
            # Maximize Sharpe ratio
            result = minimize(
                self.negative_sharpe,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif method == 'min_volatility':
            # Minimize volatility
            result = minimize(
                lambda w, r: self.calculate_portfolio_stats(w, r)['volatility'],
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif method == 'max_return':
            # Maximize return
            result = minimize(
                lambda w, r: -self.calculate_portfolio_stats(w, r)['return'],
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        if not result.success:
            # If optimization fails, return equal weights
            optimal_weights = initial_weights
        else:
            optimal_weights = result.x
        
        # Calculate optimal portfolio statistics
        stats = self.calculate_portfolio_stats(optimal_weights, returns)
        
        # Create weights dictionary
        weights_dict = {
            returns.columns[i]: optimal_weights[i] 
            for i in range(len(returns.columns))
        }
        
        return {
            'weights': weights_dict,
            'expected_return': stats['return'],
            'volatility': stats['volatility'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'optimization_success': result.success
        }
    
    def efficient_frontier(
        self, 
        returns: pd.DataFrame, 
        num_portfolios: int = 100
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            returns: Returns DataFrame
            num_portfolios: Number of portfolios to generate
        
        Returns:
            DataFrame with efficient frontier data
        """
        num_assets = len(returns.columns)
        results = []
        
        # Generate random portfolios
        np.random.seed(42)
        for _ in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            stats = self.calculate_portfolio_stats(weights, returns)
            results.append({
                'return': stats['return'],
                'volatility': stats['volatility'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'weights': weights
            })
        
        return pd.DataFrame(results)
    
    def compare_portfolios(
        self, 
        returns: pd.DataFrame, 
        methods: List[str] = None
    ) -> Dict:
        """
        Compare results from different optimization methods.
        
        Args:
            returns: Returns DataFrame
            methods: List of optimization methods
        
        Returns:
            Dictionary with comparison results
        """
        if methods is None:
            methods = ['max_sharpe', 'min_volatility', 'equal_weight']
        
        comparison = {}
        
        for method in methods:
            if method == 'equal_weight':
                # Equal weight portfolio
                num_assets = len(returns.columns)
                weights = np.array([1/num_assets] * num_assets)
                stats = self.calculate_portfolio_stats(weights, returns)
                comparison[method] = {
                    'weights': {returns.columns[i]: weights[i] for i in range(num_assets)},
                    **stats
                }
            else:
                result = self.optimize_portfolio(returns, method=method)
                comparison[method] = result
        
        return comparison
