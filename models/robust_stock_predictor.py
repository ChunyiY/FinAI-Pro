"""
Robust Stock Prediction Module for Small-Sample Data

Uses XGBoost/LightGBM with conservative hyperparameters for classification
of stock price direction under extreme data scarcity (~100 trading days).

This is a methodology demonstration for small-sample financial time-series,
NOT a live trading system.
"""
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
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


class RobustStockPredictor:
    """
    Robust stock direction classifier for small-sample data.
    
    Uses XGBoost or LightGBM with conservative hyperparameters to predict
    whether stock price will go up or down in the next 3-5 days.
    
    Key design principles:
    - Classification (not regression) to avoid overfitting
    - Conservative hyperparameters (max_depth ≤ 3, n_estimators ≤ 80)
    - Simple, interpretable features (max 10-15 features)
    - Walk-forward validation (time-series aware)
    - Strong regularization to prevent overfitting
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',  # 'xgboost' or 'lightgbm'
        forward_days: int = 5,  # Number of days ahead to predict
        transaction_cost: float = 0.001  # Estimated transaction cost (0.1%)
    ) -> None:
        """
        Initialize robust stock predictor.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            forward_days: Number of days ahead to predict (3-5 recommended)
            transaction_cost: Estimated transaction cost for label threshold
        """
        self.model_type = model_type.lower()
        self.forward_days = forward_days
        self.transaction_cost = transaction_cost
        
        # Validate model availability
        if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        if self.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create simple, interpretable features (max 10-15 features).
        
        Feature categories:
        - Returns: 1-day, 5-day log return
        - Volatility: rolling std (5-day)
        - Trend: MA5 / MA10 ratio
        - Momentum: RSI(14)
        - Volume: z-scored volume or volume change
        - Limited lags (lag ≤ 3) if necessary
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Ensure we have required columns
        if 'Close' not in df.columns:
            raise ValueError("Data must contain 'Close' column")
        
        features_df = pd.DataFrame(index=df.index)
        
        # 1. Returns (log returns are more stable than simple returns)
        if 'Close' in df.columns:
            features_df['return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
            features_df['return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
            # Lag returns (limited to 3 lags)
            features_df['return_1d_lag1'] = features_df['return_1d'].shift(1)
            features_df['return_1d_lag2'] = features_df['return_1d'].shift(2)
        
        # 2. Volatility (rolling standard deviation)
        if 'Close' in df.columns:
            features_df['volatility_5d'] = df['Close'].rolling(window=5).std()
            features_df['volatility_10d'] = df['Close'].rolling(window=10).std()
        
        # 3. Trend indicators
        if 'Close' in df.columns:
            ma5 = df['Close'].rolling(window=5).mean()
            ma10 = df['Close'].rolling(window=10).mean()
            features_df['ma5_ma10_ratio'] = ma5 / (ma10 + 1e-8)
            features_df['price_ma5_ratio'] = df['Close'] / (ma5 + 1e-8)
        
        # 4. Momentum (RSI)
        if 'Close' in df.columns:
            features_df['rsi_14'] = self._calculate_rsi(df['Close'], period=14)
        
        # 5. Volume features
        if 'Volume' in df.columns:
            # Z-scored volume (normalized)
            volume_mean = df['Volume'].rolling(window=20).mean()
            volume_std = df['Volume'].rolling(window=20).std()
            features_df['volume_zscore'] = (df['Volume'] - volume_mean) / (volume_std + 1e-8)
            features_df['volume_change'] = df['Volume'].pct_change()
        
        # 6. High-Low range (volatility proxy)
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            features_df['hl_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)
            features_df['hl_range_5d'] = features_df['hl_range'].rolling(window=5).mean()
        
        # Drop rows with NaN (from rolling windows and shifts)
        features_df = features_df.dropna()
        
        # Keep only the features we created (max 15 features)
        # Select top features if we have more than 15
        if len(features_df.columns) > 15:
            # Keep the most important ones (prioritize returns, volatility, trend)
            priority_features = [
                'return_1d', 'return_5d', 'return_1d_lag1', 'return_1d_lag2',
                'volatility_5d', 'volatility_10d',
                'ma5_ma10_ratio', 'price_ma5_ratio',
                'rsi_14',
                'volume_zscore', 'volume_change',
                'hl_range', 'hl_range_5d'
            ]
            available_priority = [f for f in priority_features if f in features_df.columns]
            features_df = features_df[available_priority]
        
        self.feature_names = list(features_df.columns)
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
    
    def _create_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Create binary classification labels.
        
        Label design: Whether future N-day return > transaction_cost
        This is more realistic than simple "up/down" as it accounts for
        transaction costs that would eat into small gains.
        
        Args:
            data: DataFrame with Close prices
            
        Returns:
            Series of binary labels (1 = positive return after costs, 0 = otherwise)
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        # Calculate forward return
        future_price = data['Close'].shift(-self.forward_days)
        forward_return = (future_price - data['Close']) / data['Close']
        
        # Label: 1 if return > transaction_cost, 0 otherwise
        # This accounts for the fact that small gains may not be profitable
        labels = (forward_return > self.transaction_cost).astype(int)
        
        return labels
    
    def prepare_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels from raw data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        # Engineer features
        features_df = self._engineer_features(data)
        
        # Create labels (align with features index)
        labels = self._create_labels(data)
        labels = labels.loc[features_df.index]
        
        # Remove rows where label is NaN (end of series)
        valid_mask = ~labels.isna()
        features_df = features_df[valid_mask]
        labels = labels[valid_mask]
        
        return features_df, labels
    
    def train(
        self,
        data: pd.DataFrame,
        validation_size: float = 0.2
    ) -> Dict:
        """
        Train the model with walk-forward validation.
        
        Uses conservative hyperparameters to prevent overfitting:
        - max_depth ≤ 3 (shallow trees)
        - n_estimators ≤ 80 (limited boosting rounds)
        - learning_rate 0.05-0.1 (moderate learning)
        - subsample < 0.8 (row sampling)
        - Strong L1/L2 regularization
        
        Args:
            data: Training data DataFrame
            validation_size: Proportion of data for validation
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare data
        features_df, labels = self.prepare_data(data)
        
        if len(features_df) < 30:
            raise ValueError(
                f"Insufficient data: got {len(features_df)} samples, "
                f"need at least 30 for meaningful training."
            )
        
        # Time-series aware split (no shuffling)
        split_idx = int(len(features_df) * (1 - validation_size))
        X_train = features_df.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_val = features_df.iloc[split_idx:]
        y_val = labels.iloc[split_idx:]
        
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
        
        # Conservative hyperparameters to prevent overfitting
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                max_depth=3,  # Shallow trees to prevent overfitting
                n_estimators=80,  # Limited boosting rounds
                learning_rate=0.08,  # Moderate learning rate
                subsample=0.75,  # Row sampling
                colsample_bytree=0.8,  # Feature sampling
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                min_child_weight=3,  # Minimum samples in leaf
                gamma=0.1,  # Minimum loss reduction
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=10,  # Early stopping in model initialization (XGBoost 2.0+)
                use_label_encoder=False
            )
        else:  # lightgbm
            self.model = lgb.LGBMClassifier(
                max_depth=3,
                n_estimators=80,
                learning_rate=0.08,
                subsample=0.75,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=5,
                random_state=42,
                verbose=-1
            )
        
        # Train model
        if self.model_type == 'xgboost':
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:  # lightgbm
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
            )
        
        self.is_trained = True
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val_scaled)
        y_val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, zero_division=0)
        recall = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        cm = confusion_matrix(y_val, y_val_pred)
        
        # Baseline comparison (always predict majority class)
        baseline_accuracy = max(y_val.mean(), 1 - y_val.mean())
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'baseline_accuracy': baseline_accuracy,
            'validation_predictions': y_val_pred,
            'validation_proba': y_val_proba,
            'validation_actual': y_val.values,
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importances_
            )) if hasattr(self.model, 'feature_importances_') else {}
        }
    
    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        train_size: float = 0.7
    ) -> Dict:
        """
        Perform walk-forward time-series cross-validation.
        
        This is time-series aware validation (no random shuffling).
        Each fold uses past data to predict future data.
        
        Args:
            data: Data DataFrame
            n_splits: Number of validation folds
            train_size: Proportion of data for initial training
            
        Returns:
            Dictionary with cross-validation results
        """
        features_df, labels = self.prepare_data(data)
        
        if len(features_df) < 50:
            raise ValueError(
                f"Insufficient data for walk-forward validation: "
                f"got {len(features_df)} samples, need at least 50."
            )
        
        total_len = len(features_df)
        initial_train_size = int(total_len * train_size)
        step_size = (total_len - initial_train_size) // n_splits
        
        if step_size < 10:
            n_splits = max(1, (total_len - initial_train_size) // 10)
            step_size = (total_len - initial_train_size) // n_splits if n_splits > 0 else 0
        
        if n_splits <= 0 or step_size <= 0:
            raise ValueError("Cannot perform walk-forward validation with available data size.")
        
        all_metrics = []
        all_predictions = []
        all_actuals = []
        
        for fold in range(n_splits):
            train_end = initial_train_size + fold * step_size
            test_end = min(train_end + step_size, total_len)
            
            if test_end - train_end < 10:
                break
            
            train_features = features_df.iloc[:train_end]
            train_labels = labels.iloc[:train_end]
            test_features = features_df.iloc[train_end:test_end]
            test_labels = labels.iloc[train_end:test_end]
            
            # Scale features
            scaler = StandardScaler()
            train_features_scaled = pd.DataFrame(
                scaler.fit_transform(train_features),
                index=train_features.index,
                columns=train_features.columns
            )
            test_features_scaled = pd.DataFrame(
                scaler.transform(test_features),
                index=test_features.index,
                columns=test_features.columns
            )
            
            # Train model for this fold
            try:
                if self.model_type == 'xgboost':
                    fold_model = xgb.XGBClassifier(
                        max_depth=3,
                        n_estimators=80,
                        learning_rate=0.08,
                        subsample=0.75,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        min_child_weight=3,
                        gamma=0.1,
                        random_state=42,
                        eval_metric='logloss',
                        early_stopping_rounds=10,  # Early stopping in model initialization (XGBoost 2.0+)
                        use_label_encoder=False
                    )
                else:  # lightgbm
                    fold_model = lgb.LGBMClassifier(
                        max_depth=3,
                        n_estimators=80,
                        learning_rate=0.08,
                        subsample=0.75,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        min_child_samples=5,
                        random_state=42,
                        verbose=-1
                    )
                
                # For XGBoost, provide eval_set for early stopping
                if self.model_type == 'xgboost':
                    fold_model.fit(
                        train_features_scaled, train_labels,
                        eval_set=[(test_features_scaled, test_labels)],
                        verbose=False
                    )
                else:  # lightgbm
                    fold_model.fit(
                        train_features_scaled, train_labels,
                        eval_set=[(test_features_scaled, test_labels)],
                        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
                    )
                
                # Predict on test set
                test_pred = fold_model.predict(test_features_scaled)
                
                # Calculate metrics
                fold_accuracy = accuracy_score(test_labels, test_pred)
                fold_precision = precision_score(test_labels, test_pred, zero_division=0)
                fold_recall = recall_score(test_labels, test_pred, zero_division=0)
                fold_f1 = f1_score(test_labels, test_pred, zero_division=0)
                
                all_metrics.append({
                    'accuracy': fold_accuracy,
                    'precision': fold_precision,
                    'recall': fold_recall,
                    'f1_score': fold_f1
                })
                all_predictions.extend(test_pred)
                all_actuals.extend(test_labels.values)
                
            except Exception as e:
                print(f"Warning: Fold {fold + 1} failed: {e}")
                continue
        
        if not all_metrics:
            raise ValueError("All validation folds failed.")
        
        # Aggregate results
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in all_metrics]),
            'n_splits': len(all_metrics),
            'predictions': np.array(all_predictions),
            'actual': np.array(all_actuals)
        }
        
        return avg_metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict stock direction for the most recent data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Array of predictions (1 = up, 0 = down)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
        
        # Prepare features
        features_df = self._engineer_features(data)
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features_df),
            index=features_df.index,
            columns=features_df.columns
        )
        
        # Predict
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of positive return.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Array of probabilities (probability of positive return)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
        
        # Prepare features
        features_df = self._engineer_features(data)
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features_df),
            index=features_df.index,
            columns=features_df.columns
        )
        
        # Predict probabilities
        proba = self.model.predict_proba(features_scaled)[:, 1]
        
        return proba

