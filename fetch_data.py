# fetch_data.py
import time
import pandas as pd
import numpy as np
from binance.client import Client
from config import API_KEY, API_SECRET, INTERVAL, SMA_SHORT_PERIOD, SMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD, HISTORICAL_DATA_PATH, BOLINGER_PERIOD, BOLINGER_STD_DEV
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='logs/data_fetch.log', format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_rsi(data, period=RSI_PERIOD):
    """Calculate RSI for the given price data."""
    close = pd.to_numeric(data['close'], errors='coerce')
    if len(close) < 2 or close.isna().all():
        return np.full(len(data), 50.0)  # Default RSI if insufficient or invalid data
    close = close.ffill().replace(0, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[:period]) if len(gain) >= period else np.mean(gain, where=~np.isnan(gain))
    avg_loss = np.mean(loss[:period]) if len(loss) >= period else np.mean(loss, where=~np.isnan(loss))
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50.0
    return np.full(len(data), rsi)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal Line for the given price data."""
    close = pd.to_numeric(data['close'], errors='coerce')
    if len(close) < max(fast, slow, signal) or close.isna().all():
        return np.zeros(len(data)), np.zeros(len(data))  # Default to zeros if insufficient or invalid data
    close = close.ffill().replace(0, np.nan)
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.values, signal_line.values

def calculate_bollinger_bands(data, period=BOLINGER_PERIOD, std_dev=BOLINGER_STD_DEV):
    """Calculate Bollinger Bands for the given price data (from paper recommendation)."""
    close = pd.to_numeric(data['close'], errors='coerce')
    if len(close) < period or close.isna().all():
        return np.full(len(data), np.nan), np.full(len(data), np.nan)  # Default to NaN if insufficient or invalid data
    close = close.ffill().replace(0, np.nan)
    sma = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band.values, lower_band.values

def calculate_atr(data, period=ATR_PERIOD):
    """Calculate Average True Range for volatility."""
    df = pd.DataFrame(data, columns=['high', 'low', 'close'])
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    if len(df) < 2 or df[['high', 'low', 'close']].isna().all().any():
        return np.full(len(df), np.nan)  # Default to NaN if insufficient or invalid data
    df = df.ffill().replace(0, np.nan)
    df['tr'] = df[['high', 'low', 'close']].apply(
        lambda row: max(row['high'] - row['low'], 
                        abs(row['high'] - df['close'].shift().fillna(df['close']).loc[row.name]), 
                        abs(row['low'] - df['close'].shift().fillna(df['close']).loc[row.name])), 
        axis=1
    )
    return df['tr'].rolling(window=period, min_periods=1).mean().values

def calculate_sma(data, short_period=SMA_SHORT_PERIOD, long_period=SMA_LONG_PERIOD):
    """Calculate Short and Long Simple Moving Averages."""
    close = pd.to_numeric(data['close'], errors='coerce')
    if len(close) < max(short_period, long_period) or close.isna().all():
        return np.full(len(data), np.nan), np.full(len(data), np.nan)  # Default to NaN if insufficient or invalid data
    close = close.ffill().replace(0, np.nan)
    sma_short = close.rolling(window=short_period, min_periods=1).mean()
    sma_long = close.rolling(window=long_period, min_periods=1).mean()
    return sma_short.values, sma_long.values

class DataFetcher:
    def __init__(self, testnet=True):
        self.client = Client(API_KEY, API_SECRET, testnet=testnet)
        self.historical_data = self.load_historical_data()

    def load_historical_data(self):
        """Load or initialize historical data from CSV, with fallback if file is missing or invalid, and validate data."""
        try:
            if os.path.exists(HISTORICAL_DATA_PATH):
                df = pd.read_csv(HISTORICAL_DATA_PATH)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp', 'close'])  # Drop rows with invalid timestamps or prices
                df = df.sort_values('timestamp')
                # Validate that the DataFrame has non-empty, non-zero data
                if not df.empty and 'symbol' in df.columns and not df['close'].isna().all() and not df['close'].eq(0).all():
                    logging.info(f"Loaded valid historical data from {HISTORICAL_DATA_PATH}")
                    return df
                else:
                    logging.warning(f"Invalid or empty historical data found at {HISTORICAL_DATA_PATH}, starting fresh")
            logging.warning(f"No valid historical data file found at {HISTORICAL_DATA_PATH}, starting fresh")
            return pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'return', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'ATR', 'SMA_Short', 'SMA_Long'])
        except Exception as e:
            logging.error(f"Error loading historical data: {e}")
            return pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'return', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'ATR', 'SMA_Short', 'SMA_Long'])

    def save_historical_data(self, df_new):
        """Save or append new historical data to CSV, ensuring no duplicates and handling invalid data."""
        if df_new.empty or df_new['close'].isna().all() or df_new['close'].eq(0).all():
            logging.warning("Attempted to save empty or invalid historical data, skipping")
            return
        if self.historical_data.empty:
            self.historical_data = df_new
        else:
            # Ensure no duplicate timestamps per symbol, keep latest
            self.historical_data = pd.concat([self.historical_data, df_new]).drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        self.historical_data = self.historical_data.sort_values('timestamp')
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(HISTORICAL_DATA_PATH), exist_ok=True)
        self.historical_data.to_csv(HISTORICAL_DATA_PATH, index=False)
        logging.info(f"Saved historical data to {HISTORICAL_DATA_PATH}")

    def get_historical_data(self, symbol, interval=INTERVAL, limit=2000, use_cache=False):  # Forced use_cache=False to bypass invalid cache
        logging.debug(f"Attempting to fetch historical data for {symbol} with interval {interval} and limit {limit}")
        try:
            if use_cache and not self.historical_data.empty:
                cached_data = self.historical_data[self.historical_data['symbol'] == symbol].copy()
                if len(cached_data) >= limit and not cached_data['close'].isna().all() and not cached_data['close'].eq(0).all():
                    cached_data = cached_data.tail(limit).reset_index(drop=True)
                    logging.debug(f"Using cached data for {symbol}")
                    return self._add_indicators(cached_data)
            else:
                logging.warning(f"Skipping invalid or empty cached data for {symbol}, fetching fresh data")

            # Split large requests into chunks of 1000 (Binance max per request)
            all_data = []
            for start in range(0, limit, 1000):
                chunk_limit = min(1000, limit - start)
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                        klines = self.client.get_klines(symbol=symbol, interval=interval, limit=chunk_limit, startTime=None if start == 0 else int(all_data[-1]['close_time'] + 1))
                        if not klines:
                            logging.error(f"No data returned for {symbol} chunk {start}-{start+chunk_limit} from Binance API - check symbol, interval, or exchange status")
                            break
                        df_chunk = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                                'close_time', 'quote_asset_volume', 'trades',
                                                                'taker_buy_base', 'taker_buy_quote', 'ignored'])
                        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms', errors='coerce')
                        df_chunk['symbol'] = symbol
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_cols:
                            df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce').replace(0, np.nan)
                        df_chunk = df_chunk.dropna(subset=['close'])
                        df_chunk['return'] = df_chunk['close'].pct_change().fillna(0)
                        all_data.extend(df_chunk.to_dict('records'))
                        break  # Exit retry loop on success
                    except Exception as e:
                        logging.error(f"Attempt {attempt + 1}/{max_retries} - Error fetching chunk {start}-{start+chunk_limit} for {symbol}: {e}")
                        if attempt == max_retries - 1:
                            return None
            if not all_data:
                logging.error(f"No data fetched for {symbol} - check API key, rate limits, or exchange status")
                return None

            df = pd.DataFrame(all_data)
            df = self._add_indicators(df)
            if not df.empty and not df['close'].isna().all() and not df['close'].eq(0).all():
                self.save_historical_data(df)
            return df
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return None

    def _add_indicators(self, df):
        """Add technical indicators to the DataFrame, handling invalid data with fallback for partial data."""
        if df.empty or 'close' not in df:
            logging.warning("Empty or invalid DataFrame in _add_indicators, returning empty")
            return df
        # Initialize with default values if data is insufficient, then fill with actual calculations
        df['RSI'] = np.full(len(df), 50.0)  # Default RSI
        df['MACD'] = np.zeros(len(df))  # Default MACD
        df['Signal_Line'] = np.zeros(len(df))  # Default Signal Line
        df['Upper_Band'] = np.full(len(df), np.nan)  # Default to NaN
        df['Lower_Band'] = np.full(len(df), np.nan)  # Default to NaN
        df['ATR'] = np.full(len(df), np.nan)  # Default to NaN
        df['SMA_Short'] = np.full(len(df), np.nan)  # Default to NaN
        df['SMA_Long'] = np.full(len(df), np.nan)  # Default to NaN

        # Calculate indicators only if data is sufficient
        close = pd.to_numeric(df['close'], errors='coerce')
        if len(close) >= 2 and not close.isna().all():
            close = close.ffill().replace(0, np.nan)
            df['RSI'] = calculate_rsi(df)
            df['MACD'], df['Signal_Line'] = calculate_macd(df)
            df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df)
            df['ATR'] = calculate_atr(df)
            df['SMA_Short'], df['SMA_Long'] = calculate_sma(df)

        # Drop rows with NaN values, but ensure at least some data is kept if possible
        df_clean = df.dropna(subset=['close'])
        if df_clean.empty:
            logging.warning("All rows dropped due to NaN values in _add_indicators, returning original df with defaults")
            return df  # Return original df with defaults if all rows are dropped

        # Replace zeros with NaN and ensure numeric precision
        df_clean = df_clean.replace(0, np.nan)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'ATR', 'SMA_Short', 'SMA_Long']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean = df_clean.dropna()

        logging.debug(f"Indicators added, final shape: {df_clean.shape}, sample close: {df_clean['close'].head()}")
        return df_clean

    def get_multi_timeframe_data(self, symbol, intervals=['1m', '5m', '15m'], limit=2000):
        """Fetch historical data for multiple timeframes."""
        data = {}
        for interval in intervals:
            df = self.get_historical_data(symbol, interval=interval, limit=limit, use_cache=False)  # Force fresh data for multi-timeframe
            if df is not None and not df.empty:
                data[interval] = df
        return data

    def get_current_price(self, symbol, retries=5):
        """Get the current ticker price for a symbol with enhanced retry logic, handling invalid data."""
        for attempt in range(retries):
            try:
                time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                price = float(self.client.get_symbol_ticker(symbol=symbol)["price"])
                if price > 0:
                    return price
                logging.warning(f"Zero price for {symbol} on attempt {attempt + 1}")
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{retries} - Error getting price for {symbol}: {e}")
                if attempt == retries - 1:
                    return None
        return None

    def get_lot_size(self, symbol):
        """Get step size for quantity adjustment."""
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            lot_size_filter = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']), None)
            return float(lot_size_filter['stepSize'])
        except Exception as e:
            logging.error(f"Error getting lot size for {symbol}: {e}")
            return None

    def get_sentiment(self, symbol):
        """Fetch sentiment for a symbol (simplified, using NewsAPI or placeholder)."""
        try:
            from sentiment import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            return analyzer.get_sentiment(symbol)
        except Exception as e:
            logging.error(f"Error fetching sentiment for {symbol}: {e}")
            return 0.0  # Default neutral sentiment