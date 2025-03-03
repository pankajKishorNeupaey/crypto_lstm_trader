# config.py
import os

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# API Keys (hardcoded for Testnet—move to .env for production)
API_KEY = "API_KEY"
API_SECRET = "API_SECRET"
NEWS_API_KEY = "NEWS_API_KEY"
TWITTER_API_KEY = "your_twitter_api_key"
TWITTER_API_SECRET = "your_twitter_api_secret"

# Trading Configuration
SYMBOLS = ["ETHUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT"]  # Ensure all symbols are listed
RUNTIME = 86400  # Runtime in seconds (24 hours for testing)
INTERVAL = "1h"  # Data interval (1 hour)
LOOKBACK = 240  # Lookback period for LSTM (4 hours at 1h intervals)
PRED_THRESHOLD = 0.1  # Prediction threshold for trades (lowered for volatile assets like SOLUSDT)
TAKE_PROFIT_BASE = 0.02  # Take-profit (2%)
STOP_LOSS_BASE = -0.01  # Stop-loss (-1%)
INR_TO_USDT_RATE = 0.011  # Conversion rate (1 USDT = 90.91 INR, updated for March 2025)
# Technical Indicator Settings
SMA_FAST_PERIOD = 9
SMA_SLOW_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
BOLINGER_PERIOD = 20
BOLINGER_STD_DEV = 2
MACD_FAST = 12  # MACD fast period
MACD_SLOW = 26  # MACD slow period
MACD_SIGNAL = 9  # MACD signal period
RISK_FREE_RATE = 0.01

# File Paths (absolute)
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
LOG_FILE = os.path.join(BASE_DIR, "logs", "training.log")
TRADING_LOG = os.path.join(BASE_DIR, "logs", "trading.log")
HISTORICAL_DATA_PATH = os.path.join(BASE_DIR, "data", "historical_data.csv")

# Trading Limits
MAX_TRADES_PER_SESSION = 5
TRADE_WINDOW_START = None  # Remove time window for 24/7 testing on Testnet
TRADE_WINDOW_END = None

# Advanced validation
if LOOKBACK <= 0:
    raise ValueError("LOOKBACK must be positive")
if INR_TO_USDT_RATE <= 0:
    raise ValueError("INR_TO_USDT_RATE must be positive")
if PRED_THRESHOLD < 0 or PRED_THRESHOLD > 1:
    raise ValueError("PRED_THRESHOLD must be between 0 and 1")
if TAKE_PROFIT_BASE <= 0 or STOP_LOSS_BASE >= 0:
    raise ValueError("TAKE_PROFIT_BASE must be positive and STOP_LOSS_BASE must be negative")
