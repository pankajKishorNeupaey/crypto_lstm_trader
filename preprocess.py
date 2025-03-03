# preprocess.py (Updated)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import LOOKBACK

class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.scaled_data = {}

    def prepare_training_data(self, df, lookback=LOOKBACK):
        """Prepare training data for LSTM, handling hourly data."""
        if df.empty or 'close' not in df.columns:
            return None, None, None
        features = ['open', 'high', 'low', 'close', 'volume', 'bid_price', 'ask_price', 'spread', 'funding_rate', 'SMA_Fast', 'SMA_Slow', 'RSI', 'ATR', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line']
        df = df.dropna(subset=features)  # Drop rows with NaN in features
        if len(df) < lookback:
            return None, None, None

        # Normalize features for hourly data
        data = df[features].values
        scaled_data = self.scaler.fit_transform(data)
        self.scaled_data[df['symbol'].iloc[0]] = self.scaler  # Store scaler for inverse transform

        X, y = self._create_sequences(scaled_data, lookback)
        return X, y, df['symbol'].iloc[0]

    def prepare_prediction_data(self, df, lookback=LOOKBACK):
        """Prepare prediction data for LSTM, handling hourly data, ensuring a valid NumPy array is returned."""
        if df.empty or 'close' not in df.columns:
            return np.array([])  # Return empty NumPy array for invalid data
        features = ['open', 'high', 'low', 'close', 'volume', 'bid_price', 'ask_price', 'spread', 'funding_rate', 'SMA_Fast', 'SMA_Slow', 'RSI', 'ATR', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line']
        df = df.dropna(subset=features)
        if len(df) < lookback:
            return np.zeros((1, lookback, len(features)))  # Return a zero-filled array for insufficient data

        data = df[features].values
        if df['symbol'].iloc[0] in self.scaled_data:
            scaled_data = self.scaled_data[df['symbol'].iloc[0]].transform(data)
        else:
            scaled_data = self.scaler.fit_transform(data)
            self.scaled_data[df['symbol'].iloc[0]] = self.scaler

        X = self._create_sequences(scaled_data, lookback)
        if isinstance(X, tuple) and len(X) == 2:  # Ensure X is not a tuple
            return X[0] if X[0].size > 0 else np.zeros((1, lookback, len(features)))  # Return X or default array
        return X if X.size > 0 else np.zeros((1, lookback, len(features)))  # Ensure X is a valid NumPy array

    def _create_sequences(self, data, lookback):
        """Create sequences for LSTM training/prediction with hourly data."""
        if len(data) < lookback + 1:
            return np.array([]), np.array([])  # Return empty arrays if data is insufficient
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback, 3])  # Use 'close' price as target (index 3 for hourly features)
        return np.array(X), np.array(y)

    def inverse_transform(self, scaled_value, symbol):
        """Inverse transform a scaled value to original scale for a specific symbol for hourly data."""
        if symbol in self.scaled_data:
            scaled_array = np.zeros((1, len(self.scaled_data[symbol].scale_)))
            scaled_array[0, 3] = scaled_value  # 'close' index
            return self.scaled_data[symbol].inverse_transform(scaled_array)[0, 3]
        return scaled_value

if __name__ == "__main__":
    from fetch_data import DataFetcher
    fetcher = DataFetcher()
    df = fetcher.get_historical_data('XRPUSDT', limit=LOOKBACK)
    preprocessor = Preprocessor()
    X, y, _ = preprocessor.prepare_training_data(df)
    if X is not None and y is not None and X.size > 0 and y.size > 0:
        print(f"Training data shape: {X.shape}, Target shape: {y.shape}")