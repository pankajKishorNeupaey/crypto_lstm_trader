# train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from fetch_data import DataFetcher
from preprocess import Preprocessor
from config import SYMBOLS, MODEL_PATH, SCALER_PATH, LOG_FILE, LOOKBACK
import logging
import os

def setup_logging():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                        format='%(asctime)s - %(message)s')

def train_lstm():
    setup_logging()
    data_fetcher = DataFetcher()
    preprocessor = Preprocessor()
    model = Sequential([
        Input(shape=(LOOKBACK, 9)),  # 9 features: price, volume, RSI, MACD, Upper/Lower Bands, ATR, SMAs
        LSTM(50, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),  # Increased dropout for generalization
        LSTM(50, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train on all symbols using historical data
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    for symbol in SYMBOLS:
        df = data_fetcher.get_historical_data(symbol, limit=2000, use_cache=False)  # Force fresh data
        if df is not None and not df.empty:
            X, y = preprocessor.prepare_training_data(df)
            if len(X) > 0 and len(y) > 0:
                model.fit(X, y, epochs=20, batch_size=32, validation_split=0.3, 
                          callbacks=[early_stopping], verbose=1)
                logging.info(f"Trained on {symbol} with {len(X)} samples")
            else:
                logging.warning(f"No valid data for training on {symbol}")
        else:
            logging.error(f"Failed to fetch data for {symbol} - check logs/data_fetch.log")

    # Save model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    preprocessor.save_scaler(SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}, Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    train_lstm()