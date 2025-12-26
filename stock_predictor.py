"""
Stock Price Predictor Module

Uses LSTM deep learning model for stock price prediction.
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


class LSTMPredictor(nn.Module):
    """LSTM neural network for stock price prediction."""
    
    def __init__(
        self, 
        input_size: int = 1, 
        hidden_size: int = 50, 
        num_layers: int = 2, 
        output_size: int = 1, 
        dropout: float = 0.2
    ) -> None:
        """
        Initialize LSTM predictor.
        
        Args:
            input_size: Size of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Size of output
            dropout: Dropout rate
        """
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            Predictions tensor
        """
        lstm_out, _ = self.lstm(x)
        # Take only the last time step output
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


class StockPredictor:
    """Stock price predictor using LSTM."""
    
    def __init__(
        self, 
        sequence_length: int = 60, 
        hidden_size: int = 50, 
        num_layers: int = 2
    ) -> None:
        """
        Initialize stock predictor.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = MinMaxScaler()
        self.model: Optional[LSTMPredictor] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data.
        
        Args:
            data: DataFrame with stock data
            target_col: Column name to predict
        
        Returns:
            Tuple of (X, y) arrays
        """
        # Extract target column
        values = data[target_col].values.reshape(-1, 1)
        
        # Normalize
        scaled_values = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_values)):
            X.append(scaled_values[i-self.sequence_length:i, 0])
            y.append(scaled_values[i, 0])
        
        return np.array(X), np.array(y)
    
    def train(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close', 
        epochs: int = 50, 
        batch_size: int = 32, 
        learning_rate: float = 0.001
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
        
        if len(X) < batch_size:
            raise ValueError("Insufficient data for training. Need more data points.")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1).to(self.device)
        
        # Create model
        self.model = LSTMPredictor(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1
        ).to(self.device)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Train
        self.model.train()
        train_losses = []
        
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
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        
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
        
        # Prepare input data
        values = data[target_col].values.reshape(-1, 1)
        scaled_values = self.scaler.transform(values)
        
        # Use last sequence_length data points as input
        last_sequence = scaled_values[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        last_sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
        
        predictions = []
        current_input = last_sequence_tensor
        
        with torch.no_grad():
            for _ in range(days):
                # Predict next value
                output = self.model(current_input)
                predictions.append(output.cpu().numpy()[0, 0])
                
                # Update input sequence (sliding window)
                new_input = torch.cat([
                    current_input[:, 1:, :],
                    output.unsqueeze(0)
                ], dim=1)
                current_input = new_input
        
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
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
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy().flatten()
        
        # Inverse transform
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        predictions_original = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, predictions_original)
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'predictions': predictions_original,
            'actual': y_test_original
        }
