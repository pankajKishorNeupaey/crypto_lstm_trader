import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from config import SYMBOLS, LOOKBACK, MODEL_PATH, SCALER_PATH, HISTORICAL_DATA_PATH
import pickle
import logging
import os
from fetch_data import DataFetcher
from preprocess import Preprocessor

# Set up logging
logging.basicConfig(filename='logs/training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_lstm():
    """Train an enhanced LSTM-CNN hybrid model with new features."""
    logger.info("Starting LSTM model training...")
    fetcher = DataFetcher()
    preprocessor = Preprocessor()
    
    all_data = pd.DataFrame()
    for symbol in SYMBOLS:
        try:
            df = fetcher.get_historical_data(symbol, limit=2000, use_cache=True)
            if df is not None and not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
                logger.info(f"Fetched data for {symbol}: {len(df)} rows")
            else:
                logger.warning(f"No data fetched for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")

    if all_data.empty:
        logger.error("No historical data available for training")
        return

    features = preprocessor.features  # Use Preprocessor's feature list
    if not all(f in all_data.columns for f in features):
        missing = set(features) - set(all_data.columns)
        logger.error(f"Missing required features: {missing}")
        return

    all_data = all_data[features].fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"Data shape before target: {all_data.shape}")

    # Prepare target (price movement: 1 if up, 0 if down/next price change)
    all_data['target'] = (all_data['close'].pct_change().shift(-1) > 0).astype(int)
    all_data = all_data.dropna()
    logger.info(f"Data shape after target: {all_data.shape}")

    if len(all_data) < LOOKBACK + 1:
        logger.error(f"Insufficient data after preprocessing: {len(all_data)} rows, need at least {LOOKBACK + 1}")
        return

    # Normalize data
    if not preprocessor.is_fitted or preprocessor.scaler.n_features_in_ != len(features):
        logger.info(f"Refitting scaler: Previous features {preprocessor.scaler.n_features_in_ if preprocessor.is_fitted else 'None'}, expected {len(features)}")
        preprocessor.fit_scaler(all_data[features].values)
    else:
        logger.info(f"Using existing scaler with {preprocessor.scaler.n_features_in_} features")
    scaled_data = preprocessor.scaler.transform(all_data[features].values)
    logger.info(f"Scaled data shape: {scaled_data.shape}")

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - LOOKBACK):
        X.append(scaled_data[i:(i + LOOKBACK)])
        y.append(all_data['target'].iloc[i + LOOKBACK])
    X, y = np.array(X), np.array(y)
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    if len(X) == 0:
        logger.error("No sequences generated for training")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Build hybrid LSTM-CNN model
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(LOOKBACK, len(features))),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, 
                        callbacks=[early_stopping], verbose=1)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Model trained - Test Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    # Save model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    preprocessor.save_scaler(SCALER_PATH)
    logger.info(f"Model and scaler saved to {MODEL_PATH} and {SCALER_PATH}")

if __name__ == "__main__":
    train_lstm()