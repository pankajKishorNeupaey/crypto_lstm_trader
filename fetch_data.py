# fetch_data.py (Updated)
from config import API_KEY, API_SECRET, INTERVAL, HISTORICAL_DATA_PATH, SMA_FAST_PERIOD, SMA_SLOW_PERIOD, RSI_PERIOD, ATR_PERIOD, BOLINGER_PERIOD, BOLINGER_STD_DEV, MACD_FAST, MACD_SLOW, MACD_SIGNAL

import ccxt
import pandas as pd
import numpy as np
import logging
import os
from dotenv import load_dotenv
import time
import websocket  # Using websocket-client for compatibility
import json
from threading import Lock, Thread
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Load environment variables
load_dotenv()

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(
    filename=os.path.join('logs', 'data_fetch.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        try:
            self.exchange = ccxt.binance({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'future', 'timeout': 30000},  # Increase timeout to 30s
            })
            self.exchange.set_sandbox_mode(True)
            balance = self.get_account_balance()
            logger.info(f"Successfully validated Binance Testnet API key. Balance: {balance} USDT")
            print(f"DataFetcher initialized. Testnet balance: {balance} USDT")
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication error: {e}. Check API_KEY and API_SECRET in .env.")
            print(f"Authentication error: {e}")
            raise ValueError("Invalid Binance Testnet API key or secret.")
        except Exception as e:
            logger.error(f"Error initializing Binance connection: {e}")
            print(f"Error initializing Binance: {e}")
            raise

        self.current_price = {}  # Use dict for multiple symbols
        self.price_lock = Lock()  # Thread-safe lock for current_price
        self.cache_timeout = 5.0  # Reduced to 5s for faster updates
        self.ws_threads = {}  # Track WebSocket threads
        self.ws_failure_count = {}  # Track WebSocket failures per symbol
        self.price_history = {}  # Track recent prices for validation
        self.last_price = {}  # Track last cached price to filter duplicates
        self.market_data_cache = {}  # Cache for market data to minimize API calls

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout)))
    def get_historical_data(self, symbol, limit=2000, use_cache=False, since=None):
        """Fetch historical OHLCV data with indicators for hourly data, with increased retries and timeout handling."""
        try:
            if use_cache and os.path.exists(HISTORICAL_DATA_PATH):
                df = pd.read_csv(HISTORICAL_DATA_PATH)
                cached_df = df[df['symbol'] == symbol].copy()
                if not cached_df.empty and len(cached_df) >= limit:
                    logger.info(f"Loaded cached data for {symbol}")
                    print(f"Loaded {len(cached_df)} rows of cached data for {symbol}")
                    return self._validate_and_enrich_data(cached_df)

            print(f"Fetching OHLCV data for {symbol} with limit={limit} since {since} (interval={INTERVAL})...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=INTERVAL, limit=limit, since=since)
            if not ohlcv:
                logger.error(f"No OHLCV data returned for {symbol}")
                print(f"Error: No OHLCV data returned for {symbol}")
                return None

            logger.info(f"Fetched {len(ohlcv)} OHLCV data points for {symbol} (requested {limit})")
            print(f"Fetched {len(ohlcv)} OHLCV data points for {symbol} (requested {limit})")
            if len(ohlcv) < limit:
                logger.warning(f"Received {len(ohlcv)} data points for {symbol}, less than requested {limit}. Proceeding anyway.")
                print(f"Warning: Only {len(ohlcv)} data points received for {symbol}, less than {limit}")

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            print(f"Initial DataFrame created for {symbol}:\n{df.tail()}")

            try:
                print(f"Fetching order book for {symbol}...")
                order_book = self.exchange.fetch_order_book(symbol, limit=10, params={'timeout': 30000})
                df['bid_price'] = order_book['bids'][0][0] if order_book['bids'] else np.nan
                df['ask_price'] = order_book['asks'][0][0] if order_book['asks'] else np.nan
                df['spread'] = df['ask_price'] - df['bid_price']
                print(f"Order book added - Bid: {df['bid_price'].iloc[-1]}, Ask: {df['ask_price'].iloc[-1]}, Spread: {df['spread'].iloc[-1]}")
            except Exception as e:
                logger.warning(f"Failed to fetch order book for {symbol}: {e}")
                print(f"Warning: Failed to fetch order book for {symbol}: {e}")
                df[['bid_price', 'ask_price', 'spread']] = np.nan

            try:
                print(f"Fetching funding rate for {symbol}...")
                funding_rate = self.exchange.fetch_funding_rate(symbol, params={'timeout': 30000})
                df['funding_rate'] = funding_rate['fundingRate'] if funding_rate else 0.0
                print(f"Funding rate added: {df['funding_rate'].iloc[-1]}")
            except Exception as e:
                logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
                print(f"Warning: Failed to fetch funding rate for {symbol}: {e}")
                df['funding_rate'] = 0.0

            print(f"Calculating indicators for {symbol}...")
            df = self._calculate_indicators(df)
            print(f"DataFrame after indicators:\n{df.tail()}")

            df = self._validate_and_enrich_data(df)
            print(f"DataFrame after validation:\n{df.tail()}")

            if os.path.exists(HISTORICAL_DATA_PATH):
                existing_df = pd.read_csv(HISTORICAL_DATA_PATH)
                existing_df = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp', 'symbol'])
            else:
                existing_df = df
            existing_df.to_csv(HISTORICAL_DATA_PATH, index=False)
            logger.info(f"Saved historical data for {symbol}")
            print(f"Saved {len(existing_df)} rows of historical data for {symbol} to {HISTORICAL_DATA_PATH}")
            return df

        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logger.error(f"Network/Exchange/Timeout error fetching data for {symbol}: {e}")
            print(f"Network/Exchange/Timeout error fetching data for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data for {symbol}: {e}")
            print(f"Unexpected error fetching historical data for {symbol}: {e}")
            return None

    def _validate_and_enrich_data(self, df):
        """Clean and validate DataFrame for hourly data."""
        if df.empty:
            logger.warning("Empty DataFrame received")
            print("Warning: Empty DataFrame received")
            return pd.DataFrame()
        df['close'] = df['close'].ffill().fillna(df['close'].mean())
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        return df

    def _calculate_indicators(self, df):
        """Calculate technical indicators with length checks for hourly data."""
        required_rows = max(SMA_SLOW_PERIOD, RSI_PERIOD, ATR_PERIOD, BOLINGER_PERIOD, MACD_SLOW)
        if len(df) < required_rows:
            logger.warning(f"Insufficient data ({len(df)} rows) for indicators. Need at least {required_rows}. Filling with NaN.")
            print(f"Warning: Insufficient data ({len(df)} rows) for indicators. Need at least {required_rows}. Filling with NaN.")
            for col in ['SMA_Fast', 'SMA_Slow', 'RSI', 'ATR', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line']:
                df[col] = np.nan
        else:
            df['SMA_Fast'] = df['close'].rolling(window=SMA_FAST_PERIOD).mean()
            df['SMA_Slow'] = df['close'].rolling(window=SMA_SLOW_PERIOD).mean()
            df['RSI'] = self.calculate_rsi(df['close'])
            df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'])
            df['Upper_Band'], df['Lower_Band'] = self.calculate_bollinger_bands(df['close'])
            df['MACD'], df['Signal_Line'] = self.calculate_macd(df['close'])
        return df

    def get_current_price(self, symbol):
        """Get real-time price via WebSocket or fallback to REST with improved reliability, longer timeout, and more retries for hourly data."""
        try:
            if symbol not in self.ws_failure_count:
                self.ws_failure_count[symbol] = 0

            # Try WebSocket first for real-time data, reducing cache reliance
            if symbol not in self.current_price or time.time() - self.current_price.get(f"{symbol}_timestamp", 0) > self.cache_timeout:
                self._setup_websocket(symbol)
                price = self.current_price.get(symbol)
                if price is None or time.time() - self.current_price.get(f"{symbol}_timestamp", 0) > self.cache_timeout:
                    raise ValueError("No recent price from WebSocket")

                # Validate against order book
                order_book = self.exchange.fetch_order_book(symbol, limit=10, params={'timeout': 30000})
                best_bid = float(order_book['bids'][0][0]) if order_book['bids'] and order_book['bids'][0][0] else None
                best_ask = float(order_book['asks'][0][0]) if order_book['asks'] and order_book['asks'][0][0] else None
                if best_bid and best_ask:
                    if not (best_bid * 0.8 <= price <= best_ask * 1.2):  # 20% deviation
                        logger.warning(f"Invalid WebSocket price for {symbol}: {price}. Using order book median.")
                        price = (best_bid + best_ask) / 2
                logger.debug(f"Current price for {symbol} (WebSocket): {price}")
                print(f"Current price for {symbol} (WebSocket): {price}")
                self._cache_price(symbol, price)
                self.ws_failure_count[symbol] = 0  # Reset failure count on success
                return price
            else:
                price = self.current_price.get(symbol)
                if time.time() - self.current_price.get(f"{symbol}_timestamp", 0) <= self.cache_timeout:
                    return price

            # Fallback to cached price only if WebSocket fails immediately
            cached_price = self._get_cached_price(symbol)
            if cached_price and time.time() - cached_price['timestamp'] < self.cache_timeout:
                price = cached_price['price']
                order_book = self.exchange.fetch_order_book(symbol, limit=10, params={'timeout': 30000})
                best_bid = float(order_book['bids'][0][0]) if order_book['bids'] and order_book['bids'][0][0] else None
                best_ask = float(order_book['asks'][0][0]) if order_book['asks'] and order_book['asks'][0][0] else None
                if best_bid and best_ask and not (best_bid * 0.8 <= price <= best_ask * 1.2):
                    logger.warning(f"Invalid cached price for {symbol}: {price}. Refreshing from REST.")
                    return self._poll_price(symbol)
                logger.debug(f"Using cached price for {symbol}: {price}")
                print(f"Using cached price for {symbol}: {price}")
                return price

        except Exception as e:
            self.ws_failure_count[symbol] = self.ws_failure_count.get(symbol, 0) + 1
            logger.error(f"WebSocket failed for {symbol} (attempt {self.ws_failure_count[symbol]}): {e}. Falling back to REST API.")
            print(f"WebSocket failed for {symbol} (attempt {self.ws_failure_count[symbol]}): {e}. Falling back to REST API.")

            # Fallback to REST with polling, longer timeout, and more retries
            backoff_time = min(2 ** (self.ws_failure_count[symbol] - 1), 20)  # Increase max backoff to 20s
            if self.ws_failure_count[symbol] > 5:  # Switch to polling after 5 failures
                logger.warning(f"Switching to REST polling for {symbol} after {self.ws_failure_count[symbol]} WebSocket failures. Waiting {backoff_time}s...")
                time.sleep(backoff_time)
                return self._poll_price(symbol)

            return self._poll_price(symbol)

    def _setup_websocket(self, symbol):
        """Set up WebSocket for real-time price updates in a thread, with increased retry attempts, robust timeout handling, and stability for hourly data."""
        if symbol in self.ws_threads and self.ws_threads[symbol].is_alive():
            return

        def websocket_thread():
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    if 'b' in data and 'a' in data and 's' in data:  # @bookTicker format: 'b' (bid), 'a' (ask), 's' (symbol)
                        mid_price = (float(data['b']) + float(data['a'])) / 2  # Use mid-price
                        with self.price_lock:
                            # Only update if price differs by more than 0.0001 (or a small threshold)
                            if symbol not in self.last_price or abs(self.last_price[symbol] - mid_price) > 0.0001:
                                self.current_price[data['s']] = mid_price
                                self.current_price[f"{data['s']}_timestamp"] = time.time()
                                self.last_price[symbol] = mid_price
                                logger.debug(f"WebSocket update: {data['s']} price = {mid_price}")
                                print(f"WebSocket update: {data['s']} price = {mid_price}")
                                self._cache_price(data['s'], mid_price)
                except Exception as e:
                    logger.error(f"Error processing WebSocket message for {symbol}: {e}")
                    ws.close()

            def on_error(ws, error):
                logger.error(f"WebSocket error for {symbol}: {error}")
                print(f"WebSocket error for {symbol}: {error}")
                ws.close()

            def on_close(ws, code, reason):
                logger.warning(f"WebSocket closed for {symbol}: {code} - {reason}")
                print(f"WebSocket closed for {symbol}: {code} - {reason}")
                # Attempt to reconnect after a delay with exponential backoff
                backoff = min(2 ** (self.ws_failure_count.get(symbol, 0) - 1), 20)  # Increase max backoff to 20s
                time.sleep(backoff)
                self._setup_websocket(symbol)  # Retry connection with robust timeout handling

            def on_open(ws):
                logger.info(f"WebSocket connected for {symbol}")
                print(f"WebSocket connected for {symbol}")
                ws.send(json.dumps({"method": "SUBSCRIBE", "params": [f"{symbol.lower()}@bookTicker"], "id": 1}))

            ws_url = "wss://testnet.binance.vision/ws"
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            try:
                # Remove unsupported 'timeout' parameter and implement a custom timeout loop
                start_time = time.time()
                while time.time() - start_time < 300:  # 5-minute timeout as a fallback
                    ws.run_forever(ping_interval=30)  # Use ping_interval for periodic checks
                    if not ws.sock or ws.sock.closed:
                        break
                    time.sleep(1)  # Small delay to prevent tight looping
                logger.warning(f"WebSocket for {symbol} timed out after 300s, attempting reconnect...")
            except Exception as e:
                logger.error(f"WebSocket thread crashed for {symbol}: {e}")
                ws.close()

        thread = Thread(target=websocket_thread, daemon=True)
        thread.start()
        self.ws_threads[symbol] = thread
        time.sleep(2)  # Give WebSocket time to connect

    def _poll_price(self, symbol):
        """Poll price via REST API with exponential backoff, longer timeout, and more retries, minimizing calls for hourly data."""
        max_attempts = 10  # Increase retries to handle Testnet instability
        for attempt in range(max_attempts):
            try:
                ticker = self.exchange.fetch_ticker(symbol, params={'timeout': 30000})  # Increase REST API timeout to 30s
                price = float(ticker['last']) if ticker['last'] is not None else None
                if price is None:
                    raise ValueError("Ticker last price is None")
                # Validate against order book, but cache to avoid repeated calls
                if symbol not in self.market_data_cache or time.time() - self.market_data_cache[symbol]['timestamp'] > 10.0:
                    order_book = self.exchange.fetch_order_book(symbol, limit=10, params={'timeout': 30000})
                    best_bid = float(order_book['bids'][0][0]) if order_book['bids'] and order_book['bids'][0][0] else None
                    best_ask = float(order_book['asks'][0][0]) if order_book['asks'] and order_book['asks'][0][0] else None
                    self.market_data_cache[symbol] = {'best_bid': best_bid, 'best_ask': best_ask, 'timestamp': time.time()}
                else:
                    best_bid = self.market_data_cache[symbol]['best_bid']
                    best_ask = self.market_data_cache[symbol]['best_ask']
                
                if best_bid and best_ask and not (best_bid * 0.8 <= price <= best_ask * 1.2):
                    raise ValueError(f"Invalid REST price for {symbol}: {price}")
                self._cache_price(symbol, price)
                logger.debug(f"REST API price for {symbol} (poll): {price}")
                print(f"REST API price for {symbol} (poll): {price}")
                return price
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                wait_time = min(2 ** attempt, 20)  # Increase max backoff to 20s
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed to poll price for {symbol}: {e}. Waiting {wait_time}s...")
                print(f"Attempt {attempt + 1}/{max_attempts} failed to poll price for {symbol}: {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error polling price for {symbol}: {e}")
                print(f"Unexpected error polling price for {symbol}: {e}")
                time.sleep(20)  # Default backoff for unexpected errors
        logger.error(f"Failed to poll price for {symbol} after {max_attempts} attempts. Returning None.")
        return None

    def _validate_price(self, symbol, price):
        """Validate price against order book and historical data, ensuring it reflects current market conditions for hourly data."""
        try:
            if symbol in self.market_data_cache and time.time() - self.market_data_cache[symbol]['timestamp'] < 10.0:
                best_bid = self.market_data_cache[symbol]['best_bid']
                best_ask = self.market_data_cache[symbol]['best_ask']
            else:
                order_book = self.exchange.fetch_order_book(symbol, limit=10, params={'timeout': 30000})
                best_bid = float(order_book['bids'][0][0]) if order_book['bids'] and order_book['bids'][0][0] else None
                best_ask = float(order_book['asks'][0][0]) if order_book['asks'] and order_book['asks'][0][0] else None
                self.market_data_cache[symbol] = {'best_bid': best_bid, 'best_ask': best_ask, 'timestamp': time.time()}

            if best_bid and best_ask:
                price_range = (best_bid * 0.8, best_ask * 1.2)  # 20% deviation
                if not (price_range[0] <= price <= price_range[1]):
                    logger.warning(f"Unrealistic price for {symbol}: {price}. Using median of bid/ask: {(best_bid + best_ask) / 2}")
                    return (best_bid + best_ask) / 2 if best_bid and best_ask else price
            return price
        except Exception as e:
            logger.warning(f"Failed to validate price for {symbol}: {e}. Returning original price: {price}")
            return price

    def _get_cached_price(self, symbol):
        """Retrieve cached price if available and validate it against recent data for hourly data."""
        cache_file = os.path.join('data', f'price_cache_{symbol}.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                if 'price' in cache and 'timestamp' in cache and time.time() - cache['timestamp'] < self.cache_timeout:
                    # Validate cached price against order book (use cached order book if recent)
                    if symbol in self.market_data_cache and time.time() - self.market_data_cache[symbol]['timestamp'] < 10.0:
                        best_bid = self.market_data_cache[symbol]['best_bid']
                        best_ask = self.market_data_cache[symbol]['best_ask']
                    else:
                        order_book = self.exchange.fetch_order_book(symbol, limit=10, params={'timeout': 30000})
                        best_bid = float(order_book['bids'][0][0]) if order_book['bids'] and order_book['bids'][0][0] else None
                        best_ask = float(order_book['asks'][0][0]) if order_book['asks'] and order_book['asks'][0][0] else None
                        self.market_data_cache[symbol] = {'best_bid': best_bid, 'best_ask': best_ask, 'timestamp': time.time()}
                    
                    if best_bid and best_ask:
                        if best_bid * 0.8 <= cache['price'] <= best_ask * 1.2:
                            return {'price': cache['price'], 'timestamp': cache['timestamp']}
                    logger.warning(f"Stale or invalid cached price for {symbol}: {cache['price']}")
        return None

    def _cache_price(self, symbol, price):
        """Cache price to file with timestamp, ensuring it’s recent and valid for hourly data."""
        cache_file = os.path.join('data', f'price_cache_{symbol}.json')
        if not os.path.exists('data'):
            os.makedirs('data')
        with open(cache_file, 'w') as f:
            json.dump({'price': price, 'timestamp': time.time()}, f)
        logger.debug(f"Cached price for {symbol}: {price}")

    def calculate_rsi(self, prices, period=RSI_PERIOD):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, highs, lows, closes, period=ATR_PERIOD):
        tr = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        tr['prev_close'] = tr['close'].shift(1)
        tr['tr'] = tr.apply(
            lambda row: max(row['high'] - row['low'],
                           abs(row['high'] - row['prev_close']) if pd.notna(row['prev_close']) else 0,
                           abs(row['low'] - row['prev_close']) if pd.notna(row['prev_close']) else 0), axis=1)
        return tr['tr'].rolling(window=period).mean()

    def calculate_bollinger_bands(self, prices, period=BOLINGER_PERIOD, std_dev=BOLINGER_STD_DEV):
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, lower_band

    def calculate_macd(self, prices):
        exp1 = prices.ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = prices.ewm(span=MACD_SLOW, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
        return macd, signal_line

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout)))
    def get_account_balance(self):
        """Fetch current USDT wallet balance from Binance Testnet with retry, longer timeout, and more robust error handling."""
        try:
            account = self.exchange.fetch_balance(params={'timeout': 30000})
            wallet_balance = float(account['USDT']['free'])  # Free balance
            total_balance = float(account['total']['USDT'])  # Total balance including unrealized P/L
            if total_balance <= 0:
                raise ValueError("No valid USDT balance available")
            logger.info(f"Fetched account balance: {total_balance} USDT (Wallet: {wallet_balance}, Total: {total_balance})")
            print(f"Fetched account balance: {total_balance} USDT (Wallet: {wallet_balance}, Total: {total_balance})")
            return total_balance
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logger.error(f"Network/Exchange/Timeout error fetching balance: {e}")
            print(f"Network/Exchange/Timeout error fetching balance: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            print(f"Error fetching balance: {e}")
            raise

if __name__ == "__main__":
    fetcher = DataFetcher()
    df = fetcher.get_historical_data('BTC/USDT')
    if df is not None:
        print("Final DataFrame:\n", df.tail())
    price = fetcher.get_current_price('BTC/USDT')
    print(f"Final current price: {price}")