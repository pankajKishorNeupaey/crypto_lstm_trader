# main.py
import os
import time
from fetch_data import DataFetcher
from preprocess import Preprocessor
from trade import TradingBot
from tensorflow.keras.models import load_model
from config import MODEL_PATH, SCALER_PATH, SYMBOLS, LOG_FILE, TRADING_LOG, INR_TO_USDT_RATE, TRADE_WINDOW_START, TRADE_WINDOW_END
from utils import Backtest, setup_logging
import logging

def main():
    # Disable oneDNN warnings for cleaner output
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Initialize logging
    try:
        logger = setup_logging(LOG_FILE)
        logger.info("Starting trading bot...")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return

    # Initialize components
    logger.info("Initializing DataFetcher...")
    try:
        data_fetcher = DataFetcher()
    except Exception as e:
        logger.error(f"Failed to initialize DataFetcher: {e}")
        return

    logger.info("Initializing Preprocessor...")
    try:
        preprocessor = Preprocessor()
    except Exception as e:
        logger.error(f"Failed to initialize Preprocessor: {e}")
        return

    logger.info("Loading model and scaler...")
    try:
        model = load_model(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    try:
        preprocessor.load_scaler(SCALER_PATH)
        if preprocessor.is_fitted and preprocessor.scaler.n_features_in_ != 13:
            logger.warning(f"Scaler expects {preprocessor.scaler.n_features_in_} features, but 13 are required. Refitting scaler...")
            os.remove(SCALER_PATH)  # Remove incompatible scaler
            preprocessor = Preprocessor()  # Reinitialize to force refit later
        logger.info(f"Loaded scaler from {SCALER_PATH} with {preprocessor.scaler.n_features_in_ if preprocessor.is_fitted else 'unfitted'} features")
    except Exception as e:
        logger.warning(f"Error loading scaler or scaler not found: {e}. Will fit new scaler during training or trading.")

    # Initialize TradingBot
    logger.info("Initializing TradingBot...")
    try:
        bot = TradingBot(data_fetcher, model, preprocessor, None)
    except Exception as e:
        logger.error(f"Failed to initialize TradingBot: {e}")
        return

    # Check initial balance
    initial_balance = bot.get_account_balance()  # Using bot's method
    if initial_balance < initial_balance * 0.9:  # Check if balance is at least 90% of itself (adjust as needed)
        logger.error(f"Insufficient balance: {initial_balance:.2f} USDT, required {initial_balance * 0.9:.2f} USDT")
        return
    logger.info(f"Initial balance: {initial_balance:.2f} USDT (~{(initial_balance / INR_TO_USDT_RATE):.2f} INR)")

    # Backtest the strategy
    logger.info("Running backtest with historical data...")
    try:
        backtest = Backtest(bot, initial_balance=initial_balance)  # Pass dynamic balance to Backtest
        profit = backtest.backtest_strategy(data_fetcher, model, preprocessor, SYMBOLS)
        logger.info(f"Backtest profit: {profit:.2f}% with {initial_balance:.2f} USDT")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        logger.info("Skipping backtest and proceeding to trading...")

    # Run trading bot (bypassing window check for testing)
    logger.info(f"Starting trading bot execution with {initial_balance:.2f} USDT (~{(initial_balance / INR_TO_USDT_RATE):.2f} INR)...")
    try:
        bot.run()
        final_balance = bot.get_account_balance()
        logger.info(f"Trading completed. Final balance: {final_balance:.2f} USDT (~{(final_balance / INR_TO_USDT_RATE):.2f} INR)")
    except Exception as e:
        logger.error(f"Trading failed: {e}")

if __name__ == "__main__":
    main()