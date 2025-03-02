# preprocess.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import LOOKBACK, HISTORICAL_DATA_PATH
import os

class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.historical_data = self.load_historical_data()
        self.is_fitted = False  # Track if scaler is fitted

    def load_historical_data(self):
        """Load historical data from CSV for preprocessing."""
        if os.path.exists(HISTORICAL_DATA_PATH):
            try:
                df = pd.read_csv(HISTORICAL_DATA_PATH)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                print(f"Error loading historical data: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def fit_scaler(self, data):
        """Fit the scaler with training data and mark as fitted."""
        if data.size > 0:  # Ensure data is not empty
            self.scaler.fit(data)
            self.is_fitted = True
        else:
            print("Warning: Empty data provided to fit_scaler, scaler not fitted")

    def prepare_training_data(self, df):
        """Prepare data for LSTM training with price, volume, RSI, MACD, Bollinger Bands, ATR, and SMAs, using historical data if available."""
        if df.empty:
            df = self.historical_data
        if df.empty or 'close' not in df or df['close'].isna().any():
            return np.array([]), np.array([])

        prices = df['close'].values.reshape(-1, 1)
        volumes = df['volume'].values.reshape(-1, 1)
        rsi = df['RSI'].values.reshape(-1, 1)
        macd = df['MACD'].values.reshape(-1, 1)
        upper_band = df['Upper_Band'].values.reshape(-1, 1)
        lower_band = df['Lower_Band'].values.reshape(-1, 1)
        atr = df['ATR'].values.reshape(-1, 1)
        sma_short = df['SMA_Short'].values.reshape(-1, 1)
        sma_long = df['SMA_Long'].values.reshape(-1, 1)

        # Scale all features together
        features = np.column_stack((prices, volumes, rsi, macd, upper_band, lower_band, atr, sma_short, sma_long))
        if not self.is_fitted:
            self.fit_scaler(features)  # Fit scaler if not already fitted
        scaled_features = self.scaler.transform(features)
        X, y = [], []
        for i in range(LOOKBACK, len(scaled_features) - 5):
            X.append(scaled_features[i - LOOKBACK:i])
            future_price = scaled_features[i + 5, 0]  # Use only price for target
            y.append(1 if future_price >= scaled_features[i - 1, 0] * 1.10 else 0)  # Target 10% gain
        return np.array(X), np.array(y)

    def prepare_prediction_data(self, df):
        """Prepare data for LSTM prediction with all features, using historical data if available."""
        if df.empty:
            df = self.historical_data.tail(LOOKBACK)
        if df.empty or 'close' not in df or len(df) < LOOKBACK:
            return np.zeros((1, LOOKBACK, 9))

        prices = df['close'].values[-LOOKBACK:].reshape(-1, 1)
        volumes = df['volume'].values[-LOOKBACK:].reshape(-1, 1)
        rsi = df['RSI'].values[-LOOKBACK:].reshape(-1, 1)
        macd = df['MACD'].values[-LOOKBACK:].reshape(-1, 1)
        upper_band = df['Upper_Band'].values[-LOOKBACK:].reshape(-1, 1)
        lower_band = df['Lower_Band'].values[-LOOKBACK:].reshape(-1, 1)
        atr = df['ATR'].values[-LOOKBACK:].reshape(-1, 1)
        sma_short = df['SMA_Short'].values[-LOOKBACK:].reshape(-1, 1)
        sma_long = df['SMA_Long'].values[-LOOKBACK:].reshape(-1, 1)

        features = np.column_stack((prices, volumes, rsi, macd, upper_band, lower_band, atr, sma_short, sma_long))
        if not self.is_fitted:
            print("Warning: MinMaxScaler not fitted, using zeros for prediction")
            return np.zeros((1, LOOKBACK, 9))  # Return zeros if scaler not fitted
        scaled_features = self.scaler.transform(features)
        return np.array([scaled_features])

    def save_scaler(self, path):
        """Save the fitted scaler."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, path):
        """Load a pre-fitted scaler and mark as fitted."""
        import pickle
        try:
            with open(path, 'rb') as f:
                self.scaler = pickle.load(f)
                self.is_fitted = True
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.is_fitted = False