# config.py
API_KEY = "your_binance_testnet_api_key"  # Replace with your Binance Testnet API key
API_SECRET = "your_binance_testnet_api_secret"  # Replace with your Binance Testnet API secret
SYMBOLS = ["DOGEUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT"]  # Focus on volatile USDT pairs
BALANCE = 12  # Starting balance in USDT (~1,000 INR at 1 INR = 0.012 USDT, verify current rate on March 1, 2025)
RUNTIME = 900  # Runtime in seconds (15 minutes for more opportunities)
INTERVAL = "1m"  # Data interval (1 minute)
LOOKBACK = 18  # Lookback period for LSTM (increased for better pattern capture)
PRED_THRESHOLD = 0.3  # Base threshold for trades, lowered for more opportunities on March 1, 2025
TAKE_PROFIT_BASE = 0.10  # Base take-profit (10%) for dynamic adjustment
STOP_LOSS_BASE = -0.03  # Base stop-loss (-3%) for dynamic adjustment
MODEL_PATH = "models/lstm_model.keras"  # Path to the LSTM model file
SCALER_PATH = "models/scaler.pkl"  # Path to the scaler file
LOG_FILE = "logs/training.log"  # Path to the log file for training
TRADING_LOG = "logs/trading.log"  # Path to the log file for trades
NEWS_API_KEY = "your_newsapi_key"  # Replace with your News API key for sentiment (optional)
INR_TO_USDT_RATE = 0.012  # Approximate conversion rate (1 INR = 0.012 USDT, verify current rate on March 1, 2025)
SMA_SHORT_PERIOD = 10  # Short SMA for trend detection
SMA_LONG_PERIOD = 50  # Long SMA for trend detection
RSI_PERIOD = 14  # RSI period for momentum
ATR_PERIOD = 14  # ATR period for volatility
RISK_FREE_RATE = 0.01  # Risk-free rate for Sharpe Ratio
HISTORICAL_DATA_PATH = "data/historical_data.csv"  # Path to store historical data
BOLINGER_PERIOD = 20  # Period for Bollinger Bands (from paper recommendation)
BOLINGER_STD_DEV = 2  # Standard deviation for Bollinger Bands