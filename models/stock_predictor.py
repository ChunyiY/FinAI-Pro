"""
Advanced Stock Price Predictor Module

Uses Bidirectional LSTM with Attention mechanism for superior stock price prediction.
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
from collections import defaultdict

# Optional imports for advanced features
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

warnings.filterwarnings('ignore')


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on important time steps."""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism.
        
        Args:
            lstm_output: LSTM output tensor (batch, seq_len, hidden_size)
        
        Returns:
            Weighted context vector
        """
        # Compute attention weights
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context


class AdvancedLSTMPredictor(nn.Module):
    """Advanced Bidirectional LSTM with Attention for stock price prediction."""
    
    def __init__(
        self, 
        input_size: int = 5,  # Multiple features: Close, Volume, High, Low, etc.
        hidden_size: int = 128, 
        num_layers: int = 3, 
        output_size: int = 1, 
        dropout: float = 0.3,
        use_attention: bool = True
    ) -> None:
        """
        Initialize advanced LSTM predictor.
        
        Args:
            input_size: Size of input features (multiple features)
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Size of output
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super(AdvancedLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
            lstm_output_size = hidden_size * 2
        else:
            lstm_output_size = hidden_size * 2
        
        # Multi-layer fully connected network with Layer Normalization
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Layer normalization for stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # Layer normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Residual connection (if input and output sizes match)
        self.use_residual = (lstm_output_size == output_size)
        if self.use_residual:
            self.residual_proj = nn.Linear(lstm_output_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            Predictions tensor
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        
        # Apply attention or use last time step
        if self.use_attention:
            context = self.attention(lstm_out)  # (batch, hidden_size * 2)
        else:
            context = lstm_out[:, -1, :]  # Take last time step
        
        # Fully connected layers
        predictions = self.fc_layers(context)
        
        # Add residual connection if applicable
        if self.use_residual:
            residual = self.residual_proj(context)
            predictions = predictions + residual
        
        return predictions


class StockPredictor:
    """Advanced stock price predictor using Bidirectional LSTM with Attention."""
    
    def __init__(
        self, 
        sequence_length: int = 60,  # Adjusted for better data efficiency
        hidden_size: int = 128,  # Larger hidden size
        num_layers: int = 3  # Deeper network
    ) -> None:
        """
        Initialize advanced stock predictor.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = RobustScaler()
        self.model: Optional[AdvancedLSTMPredictor] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = None  # Will be set based on available features
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to a DataFrame.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with engineered features added
        """
        df = df.copy()
        
        # Calculate price change features
        if 'Close' in df.columns:
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5'] = df['Close'].pct_change(periods=5)
            df['Volatility'] = df['Close'].rolling(window=10).std()
            df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Calculate volume-based features
        if 'Volume' in df.columns:
            df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-8)
            df['Volume_Change'] = df['Volume'].pct_change()
        
        # Calculate price position features (High-Low-Close relationships)
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            df['HL_Ratio'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)
            df['OC_Ratio'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-8) if 'Open' in df.columns else 0
        
        return df
    
    def _initialize_weights(self, model: nn.Module) -> None:
        """Initialize model weights for better training stability."""
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Skip LayerNorm weights (they are 1D and have default initialization)
                if 'norm' in name.lower() or 'layernorm' in name.lower():
                    continue
                
                if 'lstm' in name:
                    # LSTM weights: use orthogonal initialization
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                elif 'attention' in name or 'fc' in name or 'residual' in name:
                    # Linear layers: use Xavier/Glorot initialization
                    # Only initialize if tensor has at least 2 dimensions
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                # Initialize biases to small values
                nn.init.constant_(param, 0.0)
    
    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with multiple features.
        
        Args:
            data: DataFrame with stock data
            target_col: Column name to predict
        
        Returns:
            Tuple of (X, y) arrays
        """
        # Select features: Close, Volume, High, Low, and technical indicators if available
        feature_candidates = ['Close', 'Volume', 'High', 'Low', 'Open']
        available_features = [col for col in feature_candidates if col in data.columns]
        
        # Add technical indicators if available (more comprehensive)
        technical_indicators = [
            'MA5', 'MA10', 'MA20', 'MA50',  # Moving averages
            'RSI',  # Relative Strength Index
            'MACD', 'MACD_signal', 'MACD_hist',  # MACD components
            'BB_upper', 'BB_middle', 'BB_lower',  # Bollinger Bands
            'Volume_MA'  # Volume moving average
        ]
        for indicator in technical_indicators:
            if indicator in data.columns:
                available_features.append(indicator)
        
        # Add engineered features (price-based features)
        features_df_temp = data[available_features].copy() if available_features else data.copy()
        
        # Use helper method to add engineered features
        features_df_temp = self._add_engineered_features(features_df_temp)
        
        # Update available_features to include engineered features
        engineered_features = ['Price_Change', 'Price_Change_5', 'Volatility', 'Momentum', 
                              'Volume_Ratio', 'Volume_Change', 'HL_Ratio', 'OC_Ratio']
        for feat in engineered_features:
            if feat in features_df_temp.columns:
                available_features.append(feat)
        
        # Use the temp dataframe for further processing (it includes engineered features)
        # Ensure we have at least Close
        if 'Close' not in available_features:
            available_features = ['Close']
        
        # Extract features (only use features that exist in features_df_temp)
        existing_features = [f for f in available_features if f in features_df_temp.columns]
        if not existing_features:
            # Fallback: try to use Close from original data
            if 'Close' in data.columns:
                features_df = data[['Close']].copy()
                existing_features = ['Close']
            else:
                raise ValueError("No valid features found. Need at least 'Close' column.")
        else:
            features_df = features_df_temp[existing_features].copy()
        
        # Set feature columns
        self.feature_cols = existing_features
        available_features = existing_features
        
        # Clean NaN values: forward fill then backward fill, finally fill with 0
        features_df = features_df.ffill().bfill().fillna(0)
        
        # Drop rows that still have NaN (shouldn't happen after fillna(0), but just in case)
        features_df = features_df.dropna()
        
        if len(features_df) < self.sequence_length + 10:
            raise ValueError(
                f"Insufficient valid data after cleaning NaN values. "
                f"Got {len(features_df)} rows, need at least {self.sequence_length + 10} rows."
            )
        
        features = features_df.values
        
        # Check for any remaining NaN or infinite values
        if np.isnan(features).any() or np.isinf(features).any():
            # Replace any remaining NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        scaled_features = self.scaler.fit_transform(features)
        
        # Final check for NaN in scaled features
        if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sequences with multiple features
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_features[i, available_features.index(target_col)])
        
        return np.array(X), np.array(y)
    
    def train(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close', 
        epochs: int = 150,  # More epochs for better training
        batch_size: int = 32, 
        learning_rate: float = 0.0005  # Lower learning rate for stability
    ) -> list:
        """
        Train the model.
        
        Args:
            data: Training data DataFrame
            target_col: Target column name
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            List of training losses
        
        Raises:
            ValueError: If insufficient data for training
        """
        # Prepare data
        X, y = self.prepare_data(data, target_col)
        
        # Check if we have enough data, adjust batch size if needed
        if len(X) < batch_size:
            # If data is insufficient, reduce batch size or sequence length
            if len(X) < 10:
                raise ValueError(
                    f"Insufficient data for training. Got {len(data)} data points, "
                    f"but need at least {self.sequence_length + 10} points. "
                    f"Please select a longer data period (e.g., 2y instead of 1y)."
                )
            # Use smaller batch size if data is limited
            batch_size = min(batch_size, max(1, len(X) // 2))
            print(f"Warning: Limited data ({len(X)} samples). Using batch size: {batch_size}")
        
        # Check for NaN or infinite values before converting to tensors
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(y).any() or np.isinf(y).any():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to PyTorch tensors
        # X already has feature dimension, no need to unsqueeze
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1).to(self.device)
        
        # Final check: ensure no NaN in tensors
        if torch.isnan(X_tensor).any() or torch.isinf(X_tensor).any():
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(y_tensor).any() or torch.isinf(y_tensor).any():
            y_tensor = torch.nan_to_num(y_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Determine input size from data
        input_size = X.shape[-1]
        
        # Create advanced model
        self.model = AdvancedLSTMPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=0.3,
            use_attention=True
        ).to(self.device)
        
        # Initialize weights for better training stability
        self._initialize_weights(self.model)
        
        # Define loss and optimizer with better settings
        criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.999))
        
        # Use CosineAnnealingWarmRestarts for better learning rate scheduling
        # This helps escape local minima and improves convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=learning_rate * 0.01
        )
        
        # Also use ReduceLROnPlateau as a secondary scheduler
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, min_lr=learning_rate * 0.0001
        )
        
        # Train with early stopping
        self.model.train()
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 25  # Early stopping patience
        best_model_state = None
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_tensor))
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            epoch_loss = 0
            num_batches = 0
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
            train_losses.append(avg_loss)
            
            # Learning rate scheduling (dual schedulers)
            scheduler.step()  # CosineAnnealingWarmRestarts doesn't need loss
            plateau_scheduler.step(avg_loss)  # ReduceLROnPlateau needs loss
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        return train_losses
    
    def predict(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close', 
        days: int = 30
    ) -> np.ndarray:
        """
        Predict future prices.
        
        Args:
            data: Historical data DataFrame
            target_col: Target column name
            days: Number of days to predict
        
        Returns:
            Array of predicted prices
        
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train() first.")
        
        self.model.eval()
        
        # Prepare input data with multiple features
        if self.feature_cols is None:
            # Fallback to single feature
            values = data[target_col].values.reshape(-1, 1)
            # Clean NaN
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            scaled_values = self.scaler.transform(values)
            # Check for NaN after scaling
            scaled_values = np.nan_to_num(scaled_values, nan=0.0, posinf=0.0, neginf=0.0)
            last_sequence = scaled_values[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        else:
            # Use multiple features - need to add engineered features first
            # Get base features that we need for engineering
            base_features = ['Close', 'Volume', 'High', 'Low', 'Open']
            available_base = [f for f in base_features if f in data.columns]
            
            # Add technical indicators if available
            technical_indicators = ['MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                                  'BB_upper', 'BB_middle', 'BB_lower', 'Volume_MA']
            for indicator in technical_indicators:
                if indicator in data.columns:
                    available_base.append(indicator)
            
            # Create temp dataframe with base features
            temp_df = data[available_base].copy() if available_base else data.copy()
            
            # Add engineered features
            temp_df = self._add_engineered_features(temp_df)
            
            # Extract only the features we need (that match self.feature_cols)
            features_df = temp_df[[f for f in self.feature_cols if f in temp_df.columns]].copy()
            
            # Clean NaN
            features_df = features_df.ffill().bfill().fillna(0)
            features = features_df.values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            scaled_features = self.scaler.transform(features)
            # Check for NaN after scaling
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
            last_sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, len(self.feature_cols))
        
        last_sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
        
        predictions = []
        current_input = last_sequence_tensor
        
        with torch.no_grad():
            # Get the index of Close in feature columns (for multi-feature case)
            close_idx = 0  # Default to first feature
            if self.feature_cols is not None and 'Close' in self.feature_cols:
                close_idx = self.feature_cols.index('Close')
            
            for _ in range(days):
                # Predict next value
                output = self.model(current_input)
                pred_value = output.cpu().numpy()[0, 0]
                
                # Check for NaN in prediction
                if np.isnan(pred_value) or np.isinf(pred_value):
                    # Use last known good value or average of last sequence
                    if len(predictions) > 0:
                        pred_value = predictions[-1]
                    else:
                        pred_value = 0.0
                
                predictions.append(pred_value)
                
                # Update input sequence (sliding window)
                # Ensure output is valid before concatenating
                if torch.isnan(output).any() or torch.isinf(output).any():
                    output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Handle multi-feature vs single-feature differently
                if self.feature_cols is not None and len(self.feature_cols) > 1:
                    # Multi-feature case: need to create a full feature vector
                    # Get the last time step of current input (excluding the predicted one)
                    last_timestep = current_input[:, -1:, :].clone()  # Shape: (1, 1, num_features)
                    # Update only the Close feature with predicted value
                    # First, we need to inverse transform the prediction to get the actual scale
                    # But for simplicity, we'll use the scaled value directly
                    # Actually, we need to create a proper feature vector
                    # The simplest approach: use the last timestep's features, but update Close
                    # However, we're working with scaled values, so we need to be careful
                    
                    # Better approach: create a new feature vector by copying last timestep
                    # and updating the Close feature position
                    new_feature_vector = last_timestep.clone()
                    # Get the scaled prediction value (we need to scale it properly)
                    # For now, let's use a simpler approach: keep other features from last timestep
                    # and update Close feature
                    new_feature_vector[0, 0, close_idx] = output[0, 0]
                    
                    # Update input sequence
                    new_input = torch.cat([
                        current_input[:, 1:, :],  # Remove first timestep
                        new_feature_vector        # Add new timestep with updated Close
                    ], dim=1)
                else:
                    # Single feature case: simple concatenation
                    new_input = torch.cat([
                        current_input[:, 1:, :],
                        output.unsqueeze(0).unsqueeze(-1)  # Shape: (1, 1, 1)
                    ], dim=1)
                
                current_input = new_input
        
        # Inverse transform (need to create full feature array for inverse transform)
        if self.feature_cols is None:
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
        else:
            # For multi-feature scaler, we need to create a full feature array
            # We'll use the Close price column index
            close_idx = self.feature_cols.index(target_col) if target_col in self.feature_cols else 0
            # Create dummy array with predictions in the right position
            dummy_features = np.zeros((len(predictions), len(self.feature_cols)))
            dummy_features[:, close_idx] = predictions
            predictions = self.scaler.inverse_transform(dummy_features)[:, close_idx]
        
        return predictions
    
    def evaluate(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close', 
        test_size: float = 0.2
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            data: Data DataFrame
            target_col: Target column name
            test_size: Test set proportion
        
        Returns:
            Dictionary containing evaluation metrics
        
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train() first.")
        
        # Prepare data
        X, y = self.prepare_data(data, target_col)
        
        # Split train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # Convert to tensors (X already has feature dimension)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy().flatten()
        
        # Inverse transform
        if self.feature_cols is None:
            y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            predictions_original = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        else:
            close_idx = self.feature_cols.index(target_col) if target_col in self.feature_cols else 0
            # Create dummy arrays for inverse transform
            y_dummy = np.zeros((len(y_test), len(self.feature_cols)))
            y_dummy[:, close_idx] = y_test
            y_test_original = self.scaler.inverse_transform(y_dummy)[:, close_idx]
            
            pred_dummy = np.zeros((len(predictions), len(self.feature_cols)))
            pred_dummy[:, close_idx] = predictions
            predictions_original = self.scaler.inverse_transform(pred_dummy)[:, close_idx]
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, predictions_original)
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        
        # Calculate Direction Accuracy (predicting up/down movement)
        if len(predictions_original) > 1 and len(y_test_original) > 1:
            pred_directions = np.sign(predictions_original[1:] - predictions_original[:-1])
            actual_directions = np.sign(y_test_original[1:] - y_test_original[:-1])
            direction_accuracy = np.mean(pred_directions == actual_directions) * 100
        else:
            direction_accuracy = 0.0
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'predictions': predictions_original,
            'actual': y_test_original
        }
    
    def analyze_feature_importance(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close',
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """
        Analyze feature importance using Permutation Importance.
        
        Args:
            data: Data DataFrame
            target_col: Target column name
            n_repeats: Number of times to permute each feature
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train() first.")
        
        # Prepare data
        X, y = self.prepare_data(data, target_col)
        
        # Split train and test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Baseline score
        self.model.eval()
        with torch.no_grad():
            baseline_pred = self.model(X_test_tensor).cpu().numpy().flatten()
        baseline_mse = mean_squared_error(y_test, baseline_pred)
        
        # Permutation importance for each feature
        feature_importance = {}
        feature_names = self.feature_cols if self.feature_cols else ['Close']
        
        for feature_idx in range(X_test.shape[-1]):
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature_{feature_idx}'
            importance_scores = []
            
            for _ in range(n_repeats):
                # Permute feature
                X_test_permuted = X_test.copy()
                np.random.shuffle(X_test_permuted[:, :, feature_idx])
                
                # Predict with permuted feature
                X_test_permuted_tensor = torch.FloatTensor(X_test_permuted).to(self.device)
                with torch.no_grad():
                    permuted_pred = self.model(X_test_permuted_tensor).cpu().numpy().flatten()
                
                permuted_mse = mean_squared_error(y_test, permuted_pred)
                importance_scores.append(permuted_mse - baseline_mse)
            
            feature_importance[feature_name] = np.mean(importance_scores)
        
        # Normalize importance scores
        max_importance = max(abs(v) for v in feature_importance.values()) if feature_importance else 1.0
        if max_importance > 0:
            feature_importance = {k: v / max_importance * 100 for k, v in feature_importance.items()}
        
        return feature_importance
    
    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        target_col: str = 'Close',
        n_splits: int = 5,
        train_size: float = 0.7
    ) -> Dict:
        """
        Perform walk-forward time series cross-validation.
        
        Args:
            data: Data DataFrame
            target_col: Target column name
            n_splits: Number of validation splits
            train_size: Proportion of data for initial training
            
        Returns:
            Dictionary with validation results
        """
        if len(data) < 100:
            raise ValueError("Insufficient data for walk-forward validation. Need at least 100 data points.")
        
        total_len = len(data)
        initial_train_size = int(total_len * train_size)
        step_size = (total_len - initial_train_size) // n_splits
        
        if step_size < self.sequence_length + 10:
            n_splits = max(1, (total_len - initial_train_size) // (self.sequence_length + 10))
            step_size = (total_len - initial_train_size) // n_splits if n_splits > 0 else 0
        
        if n_splits <= 0 or step_size <= 0:
            raise ValueError("Cannot perform walk-forward validation with available data size.")
        
        all_metrics = []
        fold_predictions = []
        fold_actuals = []
        
        for fold in range(n_splits):
            train_end = initial_train_size + fold * step_size
            test_end = min(train_end + step_size, total_len)
            
            if test_end - train_end < self.sequence_length + 10:
                break
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Train model on this fold
            try:
                # Create a new model for this fold (copy current scaler settings)
                fold_model = StockPredictor(
                    sequence_length=self.sequence_length,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers
                )
                # Use fewer epochs for CV to save time
                fold_model.train(train_data, target_col=target_col, epochs=50)
                
                # Evaluate on test set (use all test data)
                # Prepare test data for evaluation
                X_test, y_test = fold_model.prepare_data(test_data, target_col)
                if len(X_test) == 0:
                    continue
                
                # Convert to tensors and predict
                X_test_tensor = torch.FloatTensor(X_test).to(fold_model.device)
                fold_model.model.eval()
                with torch.no_grad():
                    predictions = fold_model.model(X_test_tensor).cpu().numpy().flatten()
                
                # Inverse transform
                if fold_model.feature_cols is None:
                    y_test_original = fold_model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    predictions_original = fold_model.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                else:
                    close_idx = fold_model.feature_cols.index(target_col) if target_col in fold_model.feature_cols else 0
                    y_dummy = np.zeros((len(y_test), len(fold_model.feature_cols)))
                    y_dummy[:, close_idx] = y_test
                    y_test_original = fold_model.scaler.inverse_transform(y_dummy)[:, close_idx]
                    
                    pred_dummy = np.zeros((len(predictions), len(fold_model.feature_cols)))
                    pred_dummy[:, close_idx] = predictions
                    predictions_original = fold_model.scaler.inverse_transform(pred_dummy)[:, close_idx]
                
                # Calculate metrics
                mse = mean_squared_error(y_test_original, predictions_original)
                mae = mean_absolute_error(y_test_original, predictions_original)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
                
                # Direction accuracy
                if len(predictions_original) > 1 and len(y_test_original) > 1:
                    pred_directions = np.sign(predictions_original[1:] - predictions_original[:-1])
                    actual_directions = np.sign(y_test_original[1:] - y_test_original[:-1])
                    direction_accuracy = np.mean(pred_directions == actual_directions) * 100
                else:
                    direction_accuracy = 0.0
                
                fold_metrics = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'Direction_Accuracy': direction_accuracy,
                    'predictions': predictions_original,
                    'actual': y_test_original
                }
                all_metrics.append(fold_metrics)
                fold_predictions.extend(fold_metrics['predictions'])
                fold_actuals.extend(fold_metrics['actual'])
            except Exception as e:
                print(f"Warning: Fold {fold + 1} failed: {e}")
                continue
        
        if not all_metrics:
            raise ValueError("All validation folds failed.")
        
        # Aggregate results
        avg_metrics = {
            'MSE': np.mean([m['MSE'] for m in all_metrics]),
            'MAE': np.mean([m['MAE'] for m in all_metrics]),
            'RMSE': np.mean([m['RMSE'] for m in all_metrics]),
            'MAPE': np.mean([m['MAPE'] for m in all_metrics]),
            'Direction_Accuracy': np.mean([m['Direction_Accuracy'] for m in all_metrics]),
            'n_splits': len(all_metrics),
            'predictions': np.array(fold_predictions),
            'actual': np.array(fold_actuals)
        }
        
        return avg_metrics
    
    @staticmethod
    def optimize_hyperparameters(
        data: pd.DataFrame,
        target_col: str = 'Close',
        n_trials: int = 20,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            data: Training data DataFrame
            target_col: Target column name
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
            
        Returns:
            Dictionary with best hyperparameters and best score
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Please install it using: pip3 install optuna"
            )
        
        def objective(trial):
            # Check if we have enough data before starting optimization
            try:
                # Quick check: prepare data to see if we have enough samples
                temp_predictor = StockPredictor(sequence_length=30)  # Use minimum sequence length
                X_check, y_check = temp_predictor.prepare_data(data, target_col)
                
                if len(X_check) < 50:  # Minimum data requirement for optimization
                    return float('inf')
            except Exception:
                return float('inf')
            
            # Suggest hyperparameters
            sequence_length = trial.suggest_int('sequence_length', 30, 90, step=10)
            hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
            num_layers = trial.suggest_int('num_layers', 2, 4)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.2, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            try:
                # Create and train model
                predictor = StockPredictor(
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    num_layers=num_layers
                )
                
                # Train with fewer epochs for faster optimization
                predictor.train(
                    data, 
                    target_col=target_col,
                    epochs=30,  # Reduced for speed
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
                
                # Evaluate
                metrics = predictor.evaluate(data, target_col=target_col, test_size=0.2)
                
                # Return negative RMSE (Optuna minimizes, but we maximize negative RMSE)
                return -metrics['RMSE']
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')
        
        study = optuna.create_study(direction='maximize')  # Maximize negative RMSE = minimize RMSE
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        return {
            'best_params': study.best_params,
            'best_score': -study.best_value,  # Convert back to positive RMSE
            'n_trials': len(study.trials),
            'study': study
        }


class EnsembleStockPredictor:
    """
    Ensemble of multiple StockPredictor models for improved accuracy.
    Uses voting/averaging to combine predictions from multiple models.
    """
    
    def __init__(
        self,
        n_models: int = 3,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 3
    ) -> None:
        """
        Initialize ensemble predictor.
        
        Args:
            n_models: Number of models in the ensemble
            sequence_length: Length of input sequences
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
        """
        self.n_models = n_models
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.models: list[StockPredictor] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(
        self,
        data: pd.DataFrame,
        target_col: str = 'Close',
        epochs: int = 150,
        batch_size: int = 32,
        learning_rate: float = 0.0005,
        use_diverse_initialization: bool = True
    ) -> list:
        """
        Train multiple models with diversity.
        
        Args:
            data: Training data DataFrame
            target_col: Target column name
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_diverse_initialization: Whether to use different random seeds for each model
            
        Returns:
            List of training loss histories for all models
        """
        self.models = []
        all_losses = []
        
        for i in range(self.n_models):
            # Create model with potentially different initialization
            if use_diverse_initialization:
                # Set different random seeds for diversity
                torch.manual_seed(42 + i)
                np.random.seed(42 + i)
            
            model = StockPredictor(
                sequence_length=self.sequence_length,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
            
            # Train model
            losses = model.train(
                data,
                target_col=target_col,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            self.models.append(model)
            all_losses.append(losses)
        
        return all_losses
    
    def predict(
        self,
        data: pd.DataFrame,
        target_col: str = 'Close',
        days: int = 30,
        method: str = 'average'  # 'average' or 'median' or 'weighted'
    ) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            data: Historical data DataFrame
            target_col: Target column name
            days: Number of days to predict
            method: Ensemble method ('average', 'median', or 'weighted')
            
        Returns:
            Array of ensemble predictions
        """
        if not self.models:
            raise ValueError("Models not trained. Please call train() first.")
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            try:
                pred = model.predict(data, target_col=target_col, days=days)
                all_predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model prediction failed: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("All model predictions failed.")
        
        all_predictions = np.array(all_predictions)  # Shape: (n_models, days)
        
        # Combine predictions
        if method == 'average':
            ensemble_pred = np.mean(all_predictions, axis=0)
        elif method == 'median':
            ensemble_pred = np.median(all_predictions, axis=0)
        elif method == 'weighted':
            # Weight by inverse variance (models with lower variance get higher weight)
            variances = np.var(all_predictions, axis=1)
            weights = 1.0 / (variances + 1e-8)
            weights = weights / np.sum(weights)
            ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
        else:
            ensemble_pred = np.mean(all_predictions, axis=0)
        
        return ensemble_pred
    
    def evaluate(
        self,
        data: pd.DataFrame,
        target_col: str = 'Close',
        test_size: float = 0.2,
        method: str = 'average'
    ) -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            data: Data DataFrame
            target_col: Target column name
            test_size: Test set proportion
            method: Ensemble method
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.models:
            raise ValueError("Models not trained. Please call train() first.")
        
        # Get predictions from all models
        all_metrics = []
        for model in self.models:
            try:
                metrics = model.evaluate(data, target_col=target_col, test_size=test_size)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Warning: Model evaluation failed: {e}")
                continue
        
        if not all_metrics:
            raise ValueError("All model evaluations failed.")
        
        # Combine predictions
        all_predictions = np.array([m['predictions'] for m in all_metrics])
        all_actuals = all_metrics[0]['actual']  # Same for all models
        
        if method == 'average':
            ensemble_predictions = np.mean(all_predictions, axis=0)
        elif method == 'median':
            ensemble_predictions = np.median(all_predictions, axis=0)
        elif method == 'weighted':
            variances = np.var(all_predictions, axis=1)
            weights = 1.0 / (variances + 1e-8)
            weights = weights / np.sum(weights)
            ensemble_predictions = np.average(all_predictions, axis=0, weights=weights)
        else:
            ensemble_predictions = np.mean(all_predictions, axis=0)
        
        # Calculate metrics
        mse = mean_squared_error(all_actuals, ensemble_predictions)
        mae = mean_absolute_error(all_actuals, ensemble_predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((all_actuals - ensemble_predictions) / all_actuals)) * 100
        
        # Direction accuracy
        if len(ensemble_predictions) > 1 and len(all_actuals) > 1:
            pred_directions = np.sign(ensemble_predictions[1:] - ensemble_predictions[:-1])
            actual_directions = np.sign(all_actuals[1:] - all_actuals[:-1])
            direction_accuracy = np.mean(pred_directions == actual_directions) * 100
        else:
            direction_accuracy = 0.0
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'predictions': ensemble_predictions,
            'actual': all_actuals,
            'individual_metrics': all_metrics
        }
