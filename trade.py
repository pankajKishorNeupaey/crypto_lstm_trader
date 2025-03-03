# trade.py (Updated)
import time
import numpy as np
import pandas as pd
import ccxt  # Ensure ccxt is imported
from fetch_data import DataFetcher
from preprocess import Preprocessor
from config import SYMBOLS, RUNTIME, TAKE_PROFIT_BASE, STOP_LOSS_BASE, INR_TO_USDT_RATE, MAX_TRADES_PER_SESSION, LOOKBACK, PRED_THRESHOLD, TRADE_WINDOW_START, TRADE_WINDOW_END
from utils import log_trade, calculate_sharpe_ratio, calculate_var
import logging
from datetime import datetime
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='logs/trading_debug.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, data_fetcher, model, preprocessor, sentiment_analyzer=None):
        """Initialize the TradingBot with necessary components for hourly data."""
        self.sentiment_analyzer = sentiment_analyzer
        self.data_fetcher = data_fetcher
        self.model = model
        self.preprocessor = preprocessor
        self.client = data_fetcher.exchange
        try:
            self.account_balance = self.data_fetcher.get_account_balance()  # Synchronous fetch
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logger.error(f"Network/Exchange/Timeout error fetching account balance: {e}. Using default balance of 0.")
            print(f"Network/Exchange/Timeout error fetching account balance: {e}. Using default balance of 0.")
            self.account_balance = 0.0  # Fallback
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}. Using default balance of 0.")
            print(f"Error fetching account balance: {e}. Using default balance of 0.")
            self.account_balance = 0.0  # Fallback
        self.current_balance = self.account_balance  # Track remaining balance for trading
        self.trade_allocation = self.account_balance * 0.1  # Use 10% of account balance per trade (12,680.92 USDT for your balance)
        self.in_position = False
        self.buy_price = 0.0
        self.buy_quantity = 0.0  # Initialize buy quantity
        self.buy_time = None  # Initialize buy time
        self.chosen_symbol = None
        self.trades_today = 2  # Starting with 2 trades already executed
        self.open_orders = {}  # Track open orders (symbol -> order details)
        self.lot_sizes = {}
        self.price_precisions = {}
        self.market_data_cache = {}  # Cache for market data
        self.cache_timeout = 5.0  # Sync with fetch_data.py
        self._load_symbol_info()
        logger.info(f"Initialized TradingBot with account balance: {self.account_balance:.2f} USDT, trade allocation: {self.trade_allocation:.2f} USDT")
        print(f"Initialized TradingBot with account balance: {self.account_balance:.2f} USDT (~{(self.account_balance / INR_TO_USDT_RATE):.2f} INR), trade allocation: {self.trade_allocation:.2f} USDT")

    def get_account_balance(self):
        """Delegate to DataFetcher to get account balance, with error handling."""
        try:
            return self.data_fetcher.get_account_balance()
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logger.error(f"Network/Exchange/Timeout error fetching balance: {e}. Returning default balance of 0.")
            print(f"Network/Exchange/Timeout error fetching balance: {e}. Returning default balance of 0.")
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}. Returning default balance of 0.")
            print(f"Error fetching balance: {e}. Returning default balance of 0.")
            return 0.0

    def _load_symbol_info(self):
        """Load lot sizes and price precisions for all symbols."""
        try:
            markets = self.client.load_markets()
            for symbol in SYMBOLS:
                market = markets.get(symbol, {})
                self.lot_sizes[symbol] = float(market.get('info', {}).get('quantityPrecision', 0.0001))
                price_precision = market.get('precision', {}).get('price', 6)
                self.price_precisions[symbol] = -int(np.log10(price_precision)) if price_precision < 1 else price_precision
        except Exception as e:
            logger.error(f"Error loading symbol info: {e}")
            print(f"Error loading symbol info: {e}")
            for symbol in SYMBOLS:
                self.lot_sizes[symbol] = 0.0001  # Default step size
                self.price_precisions[symbol] = 6  # Default precision

    def adjust_quantity(self, symbol, price):
        """Adjust quantity to use 10–15% of account balance (1,268.09–1,902.14 USDT for your balance), ensuring full fills and matching liquidity for hourly data."""
        if price <= 0:
            logger.warning(f"Invalid price for {symbol}: {price}")
            print(f"Invalid price for {symbol}: {price}")
            return 0
        step_size = self.lot_sizes.get(symbol, 0.0001)
        precision = self.price_precisions.get(symbol, 6)
        max_position = self.trade_allocation * np.random.uniform(1.0, 1.5)  # Randomly 10–15% for diversification
        # Adjust quantity to match available liquidity in order book
        order_book = self.data_fetcher.exchange.fetch_order_book(symbol, limit=10, params={'timeout': 30000})
        best_bid = float(order_book['bids'][0][0]) if order_book['bids'] and order_book['bids'][0][0] else price
        best_ask = float(order_book['asks'][0][0]) if order_book['asks'] and order_book['asks'][0][0] else price
        quantity = max(0, min(max_position / price, max_position / price / step_size * step_size))
        # Limit quantity to available bid/ask volume to avoid partial fills
        bid_volume = float(order_book['bids'][0][1]) if order_book['bids'] else 0
        ask_volume = float(order_book['asks'][0][1]) if order_book['asks'] else 0
        if quantity > (bid_volume if 'buy' in self.open_orders.get(symbol, {}).get('side', '') else ask_volume):
            quantity = min(quantity, bid_volume if 'buy' in self.open_orders.get(symbol, {}).get('side', '') else ask_volume)
        # Cap for high-volatility assets like SOLUSDT to prevent overexposure
        if symbol == "SOLUSDT" and quantity > (max_position / best_bid):
            quantity = max(0, min(bid_volume, max_position / best_bid / step_size * step_size))
        logger.debug(f"Adjusted quantity for {symbol}: {quantity} (price={price}, max_position={max_position}, bid_volume={bid_volume}, ask_volume={ask_volume})")
        print(f"Adjusted quantity for {symbol}: {quantity} (price={price}, max_position={max_position}, bid_volume={bid_volume}, ask_volume={ask_volume})")
        return round(quantity, precision)

    @lru_cache(maxsize=128)  # Cache market data for performance
    def _fetch_market_data(self, symbol, current_price=None):
        """Fetch and cache market data with retries, longer timeout, and null checks, prioritizing profit opportunities for hourly data."""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                order_book = self.client.fetch_order_book(symbol, limit=10, params={'timeout': 30000})
                ticker = self.client.fetch_ticker(symbol, params={'timeout': 30000})
                last_price = float(ticker['last']) if ticker['last'] is not None else (current_price or 0.0)
                mark_price = float(ticker.get('markPrice', last_price)) if ticker.get('markPrice') is not None else last_price
                best_ask = float(order_book['asks'][0][0]) if order_book['asks'] and order_book['asks'][0][0] else (current_price * 1.01 if current_price else 0.0)
                best_bid = float(order_book['bids'][0][0]) if order_book['bids'] and order_book['bids'][0][0] else (current_price * 0.99 if current_price else 0.0)

                if not (best_ask and best_bid and last_price):
                    logger.warning(f"Missing market data for {symbol} on attempt {attempt + 1}: ask={best_ask}, bid={best_bid}, last={last_price}, mark={mark_price}")
                    raise ValueError("Invalid market data, retrying...")

                # Tighten price validation to 20% deviation
                price_threshold = last_price * 1.2 if last_price else (current_price * 1.2 if current_price else 0.0)
                if abs(best_ask - last_price) > price_threshold or abs(best_bid - last_price) > price_threshold:
                    logger.warning(f"Unreasonable prices for {symbol} on attempt {attempt + 1}: ask={best_ask}, bid={best_bid}, last={last_price}, current={current_price}")
                    raise ValueError("Unreasonable prices, retrying...")

                return {
                    'best_ask': best_ask,
                    'best_bid': best_bid,
                    'last_price': last_price,
                    'mark_price': mark_price
                }
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed to fetch market data for {symbol}: {e}. Retrying with exponential backoff...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch market data for {symbol} after {max_attempts} attempts: {e}")
                    return {
                        'best_ask': (current_price or 0.0) * 1.01,
                        'best_bid': (current_price or 0.0) * 0.99,
                        'last_price': current_price or 0.0,
                        'mark_price': current_price or 0.0
                    }
            except Exception as e:
                logger.error(f"Unexpected error fetching market data for {symbol}: {e}")
                time.sleep(20)  # Default backoff for unexpected errors
                return {
                    'best_ask': (current_price or 0.0) * 1.01,
                    'best_bid': (current_price or 0.0) * 0.99,
                    'last_price': current_price or 0.0,
                    'mark_price': current_price or 0.0
                }

    def place_order(self, symbol, side, quantity, order_type='limit'):
        """Place a limit order on Binance Testnet with position side, optimized for scalping to maximize 1–2% gains for hourly data, with error handling."""
        action = "BUY" if side == 'buy' else "SELL"
        position_side = 'long' if side == 'buy' else 'short'
        current_price = self.data_fetcher.get_current_price(symbol)

        try:
            if quantity <= 0:
                raise ValueError("Quantity must be greater than 0")
            if not current_price or current_price <= 0:
                raise ValueError(f"Invalid price for {symbol}: {current_price}")

            market_data = self._fetch_market_data(symbol, current_price)
            best_ask = market_data['best_ask']
            best_bid = market_data['best_bid']
            last_price = market_data['last_price']
            mark_price = market_data['mark_price']

            # Use real-time prices from WebSocket, adjusted for scalping to maximize 1–2% gains
            if side == 'buy':
                limit_price = max(best_bid * 1.001, last_price * 0.998)  # Buy slightly above bid, but below last price for profit
                # Ensure buy price is within 1% of last price to target gains
                if limit_price > last_price * 1.01:
                    limit_price = last_price * 0.998  # Cap buy price at 99.8% of last to avoid overpaying
            else:
                limit_price = min(best_ask * 0.999, last_price * 1.02)  # Sell slightly below ask, target 2% profit
                # Ensure sell price is above last price for profit, but within 2%
                if limit_price < last_price * 1.01:
                    limit_price = last_price * 1.02  # Floor sell price at 102% of last for 2% gain

            # Ensure limit price is within 2% of last price for scalping
            max_price = last_price * 1.02
            min_price = last_price * 0.98
            limit_price = min(max_price, max(limit_price, min_price))

            if limit_price <= 0:
                raise ValueError(f"Invalid limit price for {symbol}: {limit_price}")

            trade_value = limit_price * quantity
            trade_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Placing {action} limit order for {symbol}: {quantity} at {limit_price:.6f} USDT, Value: {trade_value:.2f} USDT, Max Price: {max_price:.6f}, Min Price: {min_price:.6f}")
            print(f"Placing {action} limit order for {symbol}: {quantity} at {limit_price:.6f} USDT, Value: {trade_value:.2f} USDT, Max Price: {max_price:.6f}, Min Price: {min_price:.6f}")

            order = self.client.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=quantity,
                price=limit_price,
                params={'positionSide': position_side, 'timeInForce': 'GTC', 'timeout': 30000}  # Increase timeout to 30s
            )

            # Track the order
            self.open_orders[symbol] = {
                'side': side,
                'price': limit_price,
                'quantity': quantity,
                'time': trade_time,
                'status': 'NEW'
            }
            logger.debug(f"Tracked order for {symbol}: {self.open_orders[symbol]}")

            if action == "BUY":
                if trade_value > self.current_balance:
                    raise ValueError(f"Insufficient balance: {self.current_balance:.2f} USDT, needed {trade_value:.2f}")
                self.current_balance -= trade_value
                self.in_position = True
                self.buy_price = limit_price  # Use limit price as buy price
                self.buy_quantity = quantity
                self.buy_time = trade_time
                self.chosen_symbol = symbol
                trade_details = f"Bought {quantity} {symbol} at {limit_price:.6f} USDT on {trade_time}"
                logger.info(f"Trade executed: {trade_details}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                print(f"Trade executed: {trade_details}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
            else:  # SELL
                profit_usdt = trade_value - (self.buy_price * self.buy_quantity)
                profit_inr = profit_usdt / INR_TO_USDT_RATE
                self.current_balance += trade_value
                # Sync balance with API after trade
                self.current_balance = self.get_account_balance()
                trade_details = (f"Sold {quantity} {symbol} at {limit_price:.6f} USDT on {trade_time}, "
                                f"Bought at {self.buy_price:.6f} USDT on {self.buy_time}, "
                                f"Profit/Loss: {profit_usdt:.2f} USDT (~{profit_inr:.2f} INR)")
                logger.info(f"Trade executed: {trade_details}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                print(f"Trade executed: {trade_details}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                self.in_position = False
                self.buy_price = 0.0
                self.buy_quantity = 0.0
                self.buy_time = None
                self.chosen_symbol = None
                # Update order status and manage positions
                self.open_orders[symbol] = {'status': 'FILLED', 'filled_qty': quantity}
                self._manage_positions(symbol, side)

            log_trade(symbol, action, limit_price, quantity, self.current_balance, profit=profit_usdt if action == "SELL" else None, error=None)
            return True
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            error_msg = f"Network/Exchange/Timeout: {e.__class__.__name__}: {str(e)}"
            logger.error(f"Error placing order for {symbol}: {error_msg}")
            print(f"Error placing order for {symbol}: {error_msg}")
            if symbol in self.open_orders:
                self.open_orders[symbol]['status'] = 'FAILED'
            log_trade(symbol, action, current_price or 0, quantity, self.current_balance, error=error_msg)
            return False
        except Exception as e:
            error_msg = f"{e.__class__.__name__}: {str(e)}"
            logger.error(f"Error placing order for {symbol}: {error_msg}")
            print(f"Error placing order for {symbol}: {error_msg}")
            if symbol in self.open_orders:
                self.open_orders[symbol]['status'] = 'FAILED'
            log_trade(symbol, action, current_price or 0, quantity, self.current_balance, error=error_msg)
            return False

    def _manage_positions(self, symbol, side):
        """Manage long and short positions to maximize profit and minimize losses, with error handling for hourly data."""
        try:
            positions = self.client.fetch_positions([symbol], params={'timeout': 30000})
            for position in positions:
                if position['symbol'] == symbol:
                    position_side = position['positionSide']
                    position_amt = float(position['positionAmt'])
                    unrealized_pnl = float(position['unrealizedProfit'])
                    current_price = self.data_fetcher.get_current_price(symbol)
                    if current_price <= 0:
                        continue

                    # Close position if profitable (1–2%) or loss exceeds -1%
                    if (side == 'sell' and position_side == 'LONG' and position_amt > 0) or \
                       (side == 'buy' and position_side == 'SHORT' and position_amt < 0):
                        quantity = abs(position_amt)
                        profit_pct = (unrealized_pnl / abs(float(position['notional']))) * 100
                        if profit_pct >= TAKE_PROFIT_BASE * 100 or profit_pct <= STOP_LOSS_BASE * 100:
                            if self.place_order(symbol, 'sell' if position_side == 'LONG' else 'buy', quantity):
                                logger.info(f"Closed {position_side} position for {symbol} at {current_price:.6f} USDT, Profit/Loss: {unrealized_pnl:.2f} USDT ({profit_pct:.2f}%)")
                                print(f"Closed {position_side} position for {symbol} at {current_price:.6f} USDT, Profit/Loss: {unrealized_pnl:.2f} USDT ({profit_pct:.2f}%)")
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logger.error(f"Network/Exchange/Timeout error managing positions for {symbol}: {e}")
            print(f"Network/Exchange/Timeout error managing positions for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error managing positions for {symbol}: {e}")
            print(f"Error managing positions for {symbol}: {e}")

    def check_open_orders(self, symbol):
        """Check and update open orders from Binance, ensuring consistency, with error handling and longer timeout for hourly data."""
        try:
            open_orders = self.client.fetch_open_orders(symbol, params={'timeout': 30000})
            for order in open_orders:
                order_id = order['id']
                if order['status'] in ['FILLED', 'CANCELED', 'REJECTED']:
                    if symbol in self.open_orders and self.open_orders[symbol]['status'] == 'NEW':
                        self.open_orders[symbol]['status'] = order['status']
                        logger.info(f"Updated open order for {symbol}: Status = {order['status']}")
                        print(f"Updated open order for {symbol}: Status = {order['status']}")
                # Cancel if price is unrealistic or order is stuck, using real-time price
                current_price = self.data_fetcher.get_current_price(symbol)
                if current_price and float(order['price']) > current_price * 1.2 or float(order['price']) < current_price * 0.8:
                    self.client.cancel_order(order['id'], symbol, params={'timeout': 30000})
                    logger.warning(f"Canceled order {order['id']} for {symbol} due to unrealistic price: {order['price']} vs current {current_price}")
                    print(f"Canceled order {order['id']} for {symbol} due to unrealistic price: {order['price']} vs current {current_price}")
                    if symbol in self.open_orders:
                        self.open_orders[symbol]['status'] = 'CANCELED'
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logger.warning(f"Network/Exchange/Timeout error checking open orders for {symbol}: {e}")
            print(f"Network/Exchange/Timeout error checking open orders for {symbol}: {e}")
        except Exception as e:
            logger.warning(f"Error checking open orders for {symbol}: {e}")
            print(f"Error checking open orders for {symbol}: {e}")

    def get_prediction(self, symbol):
        """Get LSTM prediction for price movement, prioritizing profitable trends for hourly data, with error handling and fixed logic."""
        try:
            df = self.data_fetcher.get_historical_data(symbol, limit=LOOKBACK, use_cache=False)
            if df is not None and not df.empty:
                X = self.preprocessor.prepare_prediction_data(df)
                if isinstance(X, np.ndarray) and X.size > 0:  # Ensure X is a valid NumPy array and not empty
                    try:
                        # Ensure X has the correct shape for prediction (batch_size, timesteps, features)
                        if X.shape[1:] != (LOOKBACK, 17):  # 17 features from config
                            logger.warning(f"Invalid shape for {symbol}: {X.shape}. Reshaping to (1, {LOOKBACK}, 17)")
                            X = X.reshape(1, LOOKBACK, 17)  # Default shape for prediction
                        prediction = self.model.predict(X, verbose=0)[0][0]
                        logger.debug(f"Prediction for {symbol}: {prediction}")
                        print(f"Prediction for {symbol}: {prediction}")
                        return max(0.0, min(1.0, prediction))
                    except Exception as e:
                        logger.error(f"Error predicting for {symbol}: {e}")
                        print(f"Error predicting for {symbol}: {e}")
                        return PRED_THRESHOLD  # Use config threshold (0.1) as default
                else:
                    logger.warning(f"Invalid prediction data for {symbol}: X is not a valid NumPy array or is empty")
                    print(f"Invalid prediction data for {symbol}: X is not a valid NumPy array or is empty")
                    return PRED_THRESHOLD  # Use config threshold (0.1) as default
            logger.warning(f"No valid data for prediction on {symbol}")
            print(f"No valid data for prediction on {symbol}")
            return PRED_THRESHOLD  # Use config threshold (0.1) as default
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logger.error(f"Network/Exchange/Timeout error fetching data for prediction on {symbol}: {e}")
            print(f"Network/Exchange/Timeout error fetching data for prediction on {symbol}: {e}")
            return PRED_THRESHOLD  # Use config threshold (0.1) as default
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            print(f"Error predicting for {symbol}: {e}")
            return PRED_THRESHOLD  # Use config threshold (0.1) as default

    def calculate_profit(self, current_price):
        """Calculate profit/loss percentage with trailing stop for scalping, targeting 1–2% gains for hourly data."""
        if self.buy_price <= 0 or current_price <= 0:
            logger.debug(f"Invalid prices for profit calc: buy_price={self.buy_price}, current_price={current_price}")
            print(f"Invalid prices for profit calc: buy_price={self.buy_price}, current_price={current_price}")
            return 0.0
        profit_pct = (current_price - self.buy_price) / self.buy_price * 100
        if profit_pct >= 0.5:  # Trailing stop at 0.5% for scalping
            self.buy_price = current_price  # Lock in gains
            profit_pct = 0.0
        return profit_pct

    def should_trade(self, current_time):
        """Check if trading is allowed within the time window for hourly data."""
        if TRADE_WINDOW_START is None or TRADE_WINDOW_END is None:
            return self.trades_today < MAX_TRADES_PER_SESSION
        trade_time = current_time.strftime("%H:%M")
        start_time = datetime.strptime(TRADE_WINDOW_START, "%H:%M").time()
        end_time = datetime.strptime(TRADE_WINDOW_END, "%H:%M").time()
        current = current_time.time()
        can_trade = (start_time <= current <= end_time) and (self.trades_today < MAX_TRADES_PER_SESSION)
        logger.debug(f"Can trade? {can_trade} (Trades today: {self.trades_today}/{MAX_TRADES_PER_SESSION}, Time: {trade_time}, Window: {TRADE_WINDOW_START}-{TRADE_WINDOW_END})")
        print(f"Can trade? {can_trade} (Trades today: {self.trades_today}/{MAX_TRADES_PER_SESSION}, Time: {trade_time}, Window: {TRADE_WINDOW_START}-{TRADE_WINDOW_END})")
        return can_trade

    def run(self):
        """Run the trading bot with scalping strategy to maximize profit for hourly data, handle network errors, and trade volatile assets."""
        logger.info(f"Starting Trading Bot: Balance {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR), Trade Allocation: {self.trade_allocation:.2f} USDT")
        print(f"Starting Trading Bot: Balance {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR), Trade Allocation: {self.trade_allocation:.2f} USDT")
        start_time = time.time()
        last_update = start_time

        logger.debug(f"Starting trading loop with RUNTIME={RUNTIME} seconds")
        print(f"Starting trading loop with RUNTIME={RUNTIME} seconds")
        iteration = 0

        while (time.time() - start_time < RUNTIME if RUNTIME is not None else False) and self.should_trade(datetime.now()):
            iteration += 1
            current_time = time.time()
            if current_time - last_update >= 60:  # Check every minute for hourly data
                logger.info(f"Iteration {iteration} - Time remaining: {RUNTIME - (current_time - start_time):.1f}s, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                print(f"Iteration {iteration} - Time remaining: {RUNTIME - (current_time - start_time):.1f}s, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                try:
                    # Check and update open orders for all symbols
                    for symbol in SYMBOLS:
                        self.check_open_orders(symbol)

                    if not self.in_position and self.trades_today < MAX_TRADES_PER_SESSION:
                        print("Checking for buy opportunities...")
                        for symbol in SYMBOLS:
                            print(f"Processing {symbol}...")
                            prediction = self.get_prediction(symbol)
                            current_price = self.data_fetcher.get_current_price(symbol)
                            if current_price is None or current_price <= 0:
                                logger.warning(f"No valid current price for {symbol}. Skipping trade.")
                                print(f"No valid current price for {symbol}. Skipping trade.")
                                continue
                            if prediction > PRED_THRESHOLD:  # Use config threshold (0.1), considering edge case
                                quantity = self.adjust_quantity(symbol, current_price)
                                if quantity > 0:
                                    # Place buy order using real-time price for profit
                                    if self.place_order(symbol, 'buy', quantity):
                                        self.trades_today += 1
                                        print(f"Trade placed successfully for {symbol}")
                                        break
                            else:
                                print(f"Prediction {prediction} below threshold {PRED_THRESHOLD}, skipping {symbol}")
                    elif self.in_position:
                        print(f"Checking sell conditions for {self.chosen_symbol}...")
                        if self.chosen_symbol:
                            current_price = self.data_fetcher.get_current_price(self.chosen_symbol)
                            if current_price is None or current_price <= 0:
                                logger.warning(f"No valid current price for {self.chosen_symbol}. Skipping sell check.")
                                print(f"No valid current price for {self.chosen_symbol}. Skipping sell check.")
                                continue
                            profit_pct = self.calculate_profit(current_price)
                            print(f"Current profit/loss for {self.chosen_symbol}: {profit_pct:.2f}% (Buy price: {self.buy_price:.6f}, Current price: {current_price:.6f})")
                            if profit_pct >= TAKE_PROFIT_BASE * 100 or profit_pct <= STOP_LOSS_BASE * 100:
                                quantity = self.adjust_quantity(self.chosen_symbol, current_price)
                                if self.place_order(self.chosen_symbol, 'sell', quantity):
                                    self.trades_today += 1
                                    print(f"Trade closed successfully for {self.chosen_symbol}")
                                    break
                            else:
                                print(f"Profit/loss {profit_pct:.2f}% not triggering sell (Take Profit: {TAKE_PROFIT_BASE*100}%, Stop Loss: {STOP_LOSS_BASE*100}%)")
                        else:
                            logger.error("No chosen symbol set while in position!")
                            print("Error: No chosen symbol set while in position!")
                            self.in_position = False  # Reset to recover
                    else:
                        print("No action: In position but no symbol set, or max trades reached")
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                    logger.error(f"Network/Exchange/Timeout error in trading loop: {e}")
                    print(f"Network/Exchange/Timeout error in trading loop: {e}")
                    time.sleep(60)  # Pause 1 minute to recover from network issues for hourly data
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    print(f"Error in trading loop: {e}")
                    time.sleep(60)  # Pause 1 minute to recover from unexpected errors for hourly data
                last_update = current_time
            else:
                logger.debug(f"Waiting for next update (time elapsed: {(current_time - last_update):.2f}s)")
            time.sleep(60)  # Wait 1 minute for hourly data checks

        if self.in_position and self.chosen_symbol:
            current_price = self.data_fetcher.get_current_price(self.chosen_symbol)
            if current_price and current_price > 0:
                quantity = self.adjust_quantity(self.chosen_symbol, current_price)
                self.place_order(self.chosen_symbol, 'sell', quantity)
                # Sync final balance with API
                self.current_balance = self.get_account_balance()
                logger.info(f"Final Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                print(f"Final Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
            else:
                logger.warning(f"No valid current price for {self.chosen_symbol}. Unable to close position.")
                print(f"No valid current price for {self.chosen_symbol}. Unable to close position.")
        else:
            # Sync final balance with API
            self.current_balance = self.get_account_balance()
            logger.info(f"No position open at end. Final Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
            print(f"No position open at end. Final Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")

if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    fetcher = DataFetcher()
    model = load_model('models/lstm_model.keras')
    preprocessor = Preprocessor()
    bot = TradingBot(fetcher, model, preprocessor, None)
    bot.run()