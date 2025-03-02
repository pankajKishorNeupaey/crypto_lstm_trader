# main.py
import os
from fetch_data import DataFetcher
from preprocess import Preprocessor
from trade import TradingBot
from tensorflow.keras.models import load_model
from config import MODEL_PATH, SCALER_PATH, SYMBOLS, LOG_FILE, TRADING_LOG, BALANCE
from sentiment import SentimentAnalyzer
from utils import backtest_strategy, setup_logging

def main():
    # Disable oneDNN warnings for cleaner output (optional)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Initialize logging for console and file
    logger = setup_logging(LOG_FILE)
    logger.info(f"Starting trading bot with {BALANCE:.2f} USDT (~{(BALANCE / 0.012):.2f} INR)...")

    # Initialize components
    print(f"Initializing DataFetcher for {BALANCE:.2f} USDT (~{(BALANCE / 0.012):.2f} INR)...")
    data_fetcher = DataFetcher()
    print("Initializing Preprocessor with historical data...")
    preprocessor = Preprocessor()
    print("Loading model and scaler...")

    try:
        model = load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        logger.error(f"Failed to load model: {e}")
        return

    try:
        preprocessor.load_scaler(SCALER_PATH)
        print(f"Loaded scaler from {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        logger.error(f"Failed to load scaler: {e}")
        return

    print("Initializing SentimentAnalyzer...")
    sentiment_analyzer = SentimentAnalyzer()

    # Backtest the strategy with detailed output, using historical data
    print("Running backtest with historical data...")
    avg_profit = 0.0  # Default to 0 if backtest fails
    try:
        for symbol in SYMBOLS:
            print(f"Backtesting {symbol} with {BALANCE:.2f} USDT...")
            df = data_fetcher.get_historical_data(symbol, limit=2000, use_cache=True)
            if df is not None and not df.empty:
                print(f"Fetched {len(df)} candles for {symbol} from historical data")
        
        avg_profit = backtest_strategy(data_fetcher, model, preprocessor, SYMBOLS)
        print(f"Backtest Average Profit: {avg_profit:.2f}% with {BALANCE:.2f} USDT")
        logger.info(f"Backtest Average Profit: {avg_profit:.2f}% with {BALANCE:.2f} USDT")
    except Exception as e:
        print(f"Backtest failed: {e}")
        logger.error(f"Backtest failed: {e}")
        print("Skipping backtest and proceeding to trading...")

    # Run trading bot
    print(f"Starting trading bot execution with {BALANCE:.2f} USDT (~{(BALANCE / 0.012):.2f} INR), using historical data...")
    bot = TradingBot(data_fetcher, model, preprocessor, sentiment_analyzer)
    bot.run()

if __name__ == "__main__":
    main()