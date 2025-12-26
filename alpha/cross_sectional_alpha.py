"""
Cross-Sectional Alpha Engine

Industry-grade alpha signal generator for multi-asset environments.
ML models are treated as noisy alpha signal generators, not predictors.

Key Design Principles:
- All learning happens in the cross-section (across stocks on same date)
- Not time-series prediction, but relative value ranking
- Output: continuous alpha scores (not classifications)
- Strong regularization to acknowledge signal noise
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None


class CrossSectionalAlphaEngine:
    """
    Cross-Sectional Alpha Engine
    
    Generates alpha signals by learning relative relationships across stocks
    on the same trading day. This is fundamentally different from time-series
    prediction - we're ranking stocks, not forecasting prices.
    
    Industry Truth:
    "We do not forecast markets. We allocate risk under uncertainty."
    
    ML outputs are treated as noisy alpha signals that require risk adjustment
    and portfolio construction to be useful.
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',  # 'xgboost' or 'lightgbm'
        max_depth: int = 4,  # Shallow trees (≤4 as per requirements)
        n_estimators: int = 100,  # Limited boosting rounds
        learning_rate: float = 0.05,  # Conservative learning rate
        reg_alpha: float = 0.1,  # L1 regularization
        reg_lambda: float = 1.0,  # L2 regularization
        subsample: float = 0.8,  # Row sampling
        colsample_bytree: float = 0.8  # Feature sampling
    ) -> None:
        """
        Initialize cross-sectional alpha engine.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            max_depth: Maximum tree depth (≤4 for shallow trees)
            n_estimators: Number of boosting rounds (limited)
            learning_rate: Learning rate (conservative)
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            subsample: Row sampling ratio
            colsample_bytree: Feature sampling ratio
        """
        self.model_type = model_type.lower()
        self.max_depth = min(max_depth, 4)  # Enforce ≤4 constraint
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
        # Validate model availability
        if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        if self.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def _engineer_features(self, panel_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for cross-sectional analysis.
        
        Features are computed per-stock in time dimension, then used
        in cross-section (across stocks on same date).
        
        Feature categories (reusing existing logic where possible):
        - Returns: 1-day, 5-day, 10-day log returns
        - Volatility: rolling std (5-day, 20-day)
        - Trend: MA ratios (MA5/MA10, MA10/MA20, price/MA ratios)
        - Momentum: RSI(14)
        - Volume: z-scored volume, volume change
        - Range: High-Low range normalized by price
        
        Args:
            panel_data: Panel data with columns [date, ticker, ...features...]
            
        Returns:
            DataFrame with engineered features (same panel structure)
        """
        # Ensure panel structure: date | ticker | features
        if 'date' not in panel_data.columns or 'ticker' not in panel_data.columns:
            raise ValueError("Panel data must have 'date' and 'ticker' columns")
        
        df = panel_data.copy()
        features_df = pd.DataFrame(index=df.index)
        
        # Group by ticker to compute time-series features
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()
            ticker_data = ticker_data.sort_values('date')
            
            # Ensure we have required columns
            if 'Close' not in ticker_data.columns:
                # If Close not available, try to use price column
                if 'price' in ticker_data.columns:
                    ticker_data['Close'] = ticker_data['price']
                else:
                    continue
            
            # 1. Returns (log returns for stability)
            if 'Close' in ticker_data.columns:
                ticker_data['return_1d'] = np.log(ticker_data['Close'] / ticker_data['Close'].shift(1))
                ticker_data['return_5d'] = np.log(ticker_data['Close'] / ticker_data['Close'].shift(5))
                ticker_data['return_10d'] = np.log(ticker_data['Close'] / ticker_data['Close'].shift(10))
            
            # 2. Volatility (rolling standard deviation)
            if 'Close' in ticker_data.columns:
                ticker_data['volatility_5d'] = ticker_data['Close'].rolling(window=5).std()
                ticker_data['volatility_20d'] = ticker_data['Close'].rolling(window=20).std()
            
            # 3. Trend indicators (moving averages)
            if 'Close' in ticker_data.columns:
                ma5 = ticker_data['Close'].rolling(window=5).mean()
                ma10 = ticker_data['Close'].rolling(window=10).mean()
                ma20 = ticker_data['Close'].rolling(window=20).mean()
                
                ticker_data['ma5_ma10_ratio'] = ma5 / (ma10 + 1e-8)
                ticker_data['ma10_ma20_ratio'] = ma10 / (ma20 + 1e-8)
                ticker_data['price_ma5_ratio'] = ticker_data['Close'] / (ma5 + 1e-8)
                ticker_data['price_ma20_ratio'] = ticker_data['Close'] / (ma20 + 1e-8)
            
            # 4. Momentum (RSI)
            if 'Close' in ticker_data.columns:
                ticker_data['rsi_14'] = self._calculate_rsi(ticker_data['Close'], period=14)
            
            # 5. Volume features
            if 'Volume' in ticker_data.columns:
                volume_mean = ticker_data['Volume'].rolling(window=20).mean()
                volume_std = ticker_data['Volume'].rolling(window=20).std()
                ticker_data['volume_zscore'] = (ticker_data['Volume'] - volume_mean) / (volume_std + 1e-8)
                ticker_data['volume_change'] = ticker_data['Volume'].pct_change()
            
            # 6. High-Low range (volatility proxy)
            if all(col in ticker_data.columns for col in ['High', 'Low', 'Close']):
                ticker_data['hl_range'] = (ticker_data['High'] - ticker_data['Low']) / (ticker_data['Close'] + 1e-8)
                ticker_data['hl_range_5d'] = ticker_data['hl_range'].rolling(window=5).mean()
            
            # Update features_df with ticker-specific features
            # Reindex ticker_data to match original df index
            ticker_data_reindexed = ticker_data.reindex(df[ticker_mask].index)
            
            feature_cols = [
                'return_1d', 'return_5d', 'return_10d',
                'volatility_5d', 'volatility_20d',
                'ma5_ma10_ratio', 'ma10_ma20_ratio', 'price_ma5_ratio', 'price_ma20_ratio',
                'rsi_14',
                'volume_zscore', 'volume_change',
                'hl_range', 'hl_range_5d'
            ]
            
            for col in feature_cols:
                if col in ticker_data_reindexed.columns:
                    if col not in features_df.columns:
                        features_df[col] = np.nan
                    features_df.loc[ticker_mask, col] = ticker_data_reindexed[col].values
        
        # Keep date and ticker columns
        features_df['date'] = df['date']
        features_df['ticker'] = df['ticker']
        
        # Drop rows with NaN (from rolling windows)
        features_df = features_df.dropna()
        
        # Store feature names (excluding date and ticker)
        self.feature_names = [col for col in features_df.columns if col not in ['date', 'ticker']]
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: Series of closing prices
            period: Period for RSI calculation
            
        Returns:
            Series containing RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_cross_sectional_labels(
        self,
        panel_data: pd.DataFrame,
        forward_days: int = 5
    ) -> pd.Series:
        """
        Create cross-sectional excess return labels.
        
        Label definition (for training only):
        label_i(t) = future_return_i(t, N days) - mean_future_return_all_stocks(t, N days)
        
        This measures relative outperformance within the same trading day.
        
        Args:
            panel_data: Panel data with date, ticker, Close columns
            forward_days: Number of days ahead for future return
            
        Returns:
            Series of cross-sectional excess returns
        """
        if 'date' not in panel_data.columns or 'ticker' not in panel_data.columns:
            raise ValueError("Panel data must have 'date' and 'ticker' columns")
        
        if 'Close' not in panel_data.columns:
            if 'price' in panel_data.columns:
                panel_data['Close'] = panel_data['price']
            else:
                raise ValueError("Panel data must have 'Close' or 'price' column")
        
        df = panel_data.copy()
        labels = pd.Series(index=df.index, dtype=float)
        
        # For each date, calculate cross-sectional labels
        for date in df['date'].unique():
            date_mask = df['date'] == date
            date_data = df[date_mask].copy()
            
            # Calculate future returns for each stock on this date
            future_returns = []
            valid_indices = []
            
            for idx, row in date_data.iterrows():
                ticker = row['ticker']
                current_price = row['Close']
                
                # Find future price (forward_days ahead)
                ticker_data = df[df['ticker'] == ticker].sort_values('date')
                current_idx = ticker_data.index.get_loc(idx)
                
                if current_idx + forward_days < len(ticker_data):
                    future_price = ticker_data.iloc[current_idx + forward_days]['Close']
                    future_return = (future_price - current_price) / current_price
                    future_returns.append(future_return)
                    valid_indices.append(idx)
            
            if len(future_returns) > 0:
                # Calculate mean future return across all stocks on this date
                mean_future_return = np.mean(future_returns)
                
                # Label = individual return - mean return (cross-sectional excess)
                for i, idx in enumerate(valid_indices):
                    labels.loc[idx] = future_returns[i] - mean_future_return
        
        return labels
    
    def prepare_panel_data(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        forward_days: int = 5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare panel data from multiple stock time-series.
        
        Converts time-series data for multiple stocks into panel format:
        date | ticker | feature_1 | feature_2 | ... | feature_k
        
        Args:
            stocks_data: Dictionary mapping ticker to DataFrame with OHLCV data
            forward_days: Number of days ahead for label calculation
            
        Returns:
            Tuple of (features_panel, labels_series)
        """
        # Convert to panel format
        panel_list = []
        
        for ticker, data in stocks_data.items():
            if data.empty:
                continue
            
            # Ensure date is in index or as column
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                data['date'] = data.index
            elif 'date' not in data.columns:
                raise ValueError(f"Data for {ticker} must have date index or 'date' column")
            
            data['ticker'] = ticker
            panel_list.append(data.reset_index(drop=True))
        
        if not panel_list:
            raise ValueError("No valid stock data provided")
        
        # Combine into panel
        panel_data = pd.concat(panel_list, ignore_index=True)
        panel_data = panel_data.sort_values(['date', 'ticker'])
        
        # Engineer features
        features_panel = self._engineer_features(panel_data)
        
        # Create labels
        labels = self._create_cross_sectional_labels(panel_data, forward_days=forward_days)
        
        # Align labels with features
        labels = labels.loc[features_panel.index]
        
        # Remove rows where label is NaN
        valid_mask = ~labels.isna()
        features_panel = features_panel[valid_mask]
        labels = labels[valid_mask]
        
        return features_panel, labels
    
    def train(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        forward_days: int = 5,
        validation_size: float = 0.2
    ) -> Dict:
        """
        Train cross-sectional alpha model.
        
        Uses time-aware walk-forward validation:
        - Train: dates ≤ T
        - Validate: dates > T
        
        Args:
            stocks_data: Dictionary mapping ticker to DataFrame
            forward_days: Number of days ahead for label calculation
            validation_size: Proportion of dates for validation
            
        Returns:
            Dictionary with training results
        """
        # Prepare panel data
        features_panel, labels = self.prepare_panel_data(stocks_data, forward_days=forward_days)
        
        if len(features_panel) < 100:
            raise ValueError(
                f"Insufficient data: got {len(features_panel)} samples, "
                f"need at least 100 for meaningful cross-sectional training."
            )
        
        # Time-aware split (by date, not random)
        unique_dates = sorted(features_panel['date'].unique())
        split_idx = int(len(unique_dates) * (1 - validation_size))
        train_dates = set(unique_dates[:split_idx])
        val_dates = set(unique_dates[split_idx:])
        
        train_mask = features_panel['date'].isin(train_dates)
        val_mask = features_panel['date'].isin(val_dates)
        
        X_train = features_panel[train_mask][self.feature_names]
        y_train = labels[train_mask]
        X_val = features_panel[val_mask][self.feature_names]
        y_val = labels[val_mask]
        
        if len(X_train) < 50 or len(X_val) < 20:
            raise ValueError(
                f"Insufficient data after split: train={len(X_train)}, val={len(X_val)}"
            )
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            index=X_val.index,
            columns=X_val.columns
        )
        
        # Initialize model with strong regularization
        # Note: Alpha signals are inherently noisy - Sharpe expected to be low
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                eval_metric='rmse'
            )
        else:  # lightgbm
            self.model = lgb.LGBMRegressor(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_child_samples=5,
                random_state=42,
                verbose=-1
            )
        
        # Train model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        self.is_trained = True
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        from scipy.stats import spearmanr
        
        # Rank IC (Spearman correlation between predicted alpha and actual excess return)
        rank_ic, _ = spearmanr(y_val_pred, y_val)
        
        # Mean squared error
        mse = np.mean((y_val_pred - y_val) ** 2)
        rmse = np.sqrt(mse)
        
        # Mean absolute error
        mae = np.mean(np.abs(y_val_pred - y_val))
        
        return {
            'rank_ic': rank_ic,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'validation_predictions': y_val_pred,
            'validation_actual': y_val.values,
            'validation_dates': features_panel[val_mask]['date'].values,
            'validation_tickers': features_panel[val_mask]['ticker'].values,
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importances_
            )) if hasattr(self.model, 'feature_importances_') else {}
        }
    
    def predict_alpha(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        target_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Predict raw alpha scores for stocks on a given date.
        
        Returns raw alpha scores (alpha_raw_i) that require risk adjustment.
        These are NOT portfolio weights - they are noisy signals.
        
        Args:
            stocks_data: Dictionary mapping ticker to DataFrame
            target_date: Target date for prediction (default: most recent date)
            
        Returns:
            DataFrame with columns: date, ticker, alpha_raw
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
        
        # Prepare panel data for target date
        panel_list = []
        
        for ticker, data in stocks_data.items():
            if data.empty:
                continue
            
            # Ensure date is in index or as column
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                data['date'] = data.index
            elif 'date' not in data.columns:
                raise ValueError(f"Data for {ticker} must have date index or 'date' column")
            
            data['ticker'] = ticker
            panel_list.append(data.reset_index(drop=True))
        
        if not panel_list:
            raise ValueError("No valid stock data provided")
        
        panel_data = pd.concat(panel_list, ignore_index=True)
        panel_data = panel_data.sort_values(['date', 'ticker'])
        
        # Engineer features
        features_panel = self._engineer_features(panel_data)
        
        # Select target date (most recent if not specified)
        if target_date is None:
            target_date = features_panel['date'].max()
        
        target_mask = features_panel['date'] == target_date
        
        if not target_mask.any():
            raise ValueError(f"No data available for target date: {target_date}")
        
        target_features = features_panel[target_mask][self.feature_names]
        
        # Scale features
        target_features_scaled = pd.DataFrame(
            self.scaler.transform(target_features),
            index=target_features.index,
            columns=target_features.columns
        )
        
        # Predict raw alpha scores
        alpha_raw = self.model.predict(target_features_scaled)
        
        # Return results
        results = pd.DataFrame({
            'date': features_panel[target_mask]['date'].values,
            'ticker': features_panel[target_mask]['ticker'].values,
            'alpha_raw': alpha_raw
        })
        
        return results

