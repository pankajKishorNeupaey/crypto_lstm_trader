# trade.py
import time
import numpy as np
import pandas as pd
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
from fetch_data import DataFetcher
from preprocess import Preprocessor
from config import SYMBOLS, BALANCE, RUNTIME, PRED_THRESHOLD, TAKE_PROFIT_BASE, STOP_LOSS_BASE, INR_TO_USDT_RATE, SMA_SHORT_PERIOD, SMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD, LOOKBACK
from utils import log_trade, calculate_sharpe_ratio, calculate_var
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='logs/trading_debug.log', format='%(asctime)s - %(levelname)s - %(message)s')

class TradingBot:
    def __init__(self, data_fetcher, model, preprocessor, sentiment_analyzer):
        """Initialize the TradingBot with necessary components."""
        self.sentiment_analyzer = sentiment_analyzer   
        self.data_fetcher = data_fetcher
        self.model = model
        self.preprocessor = preprocessor
        self.client = data_fetcher.client
        self.current_balance = BALANCE  # Track balance in USDT
        self.in_position = False
        self.buy_price = 0.0  # Initialize to 0.0, updated on buy
        self.chosen_symbol = None
        self.strategy = 'momentum'  # Default strategy

    def adjust_quantity(self, symbol, price, atr=None):
        """Adjust quantity to match Binance lot size, precision, and balance, prioritizing low-priced pairs with volatility weighting."""
        if price <= 0:
            logging.warning(f"Invalid price for {symbol}: {price}")
            return 0
        step_size = self.data_fetcher.get_lot_size(symbol)
        if not step_size or step_size <= 0:
            logging.warning(f"No valid step size for {symbol}")
            return 0
        precision = int(-np.log10(step_size)) if step_size < 1 else 0
        # Use ATR for volatility-based sizing (default to 0.0001 if not provided)
        atr = atr if atr and atr > 0 else 0.0001
        volatility_factor = max(0.2, min(1.0, 1 / (atr * 1000)))  # Normalize ATR (e.g., *1000 for small values)
        # Use 50%-90% of balance based on volatility, capped for low-priced pairs
        max_quantity = (self.current_balance * (0.5 + 0.4 * volatility_factor)) / price
        if symbol in ["SHIBUSDT", "DOGEUSDT"]:
            max_quantity = min(max_quantity, 1000000 / step_size)  # Cap for low-priced pairs
        adjusted = max(0, min(round(max_quantity / step_size) * step_size, max_quantity))
        return round(adjusted, precision)

    def place_order(self, symbol, side, quantity):
        """Place a market order on Binance Testnet, updating balance and ensuring capital protection."""
        action = "UNKNOWN"
        try:
            time.sleep(0.1)  # Ensure fresh data
            if quantity <= 0:
                raise ValueError("Quantity must be greater than 0")
            action = "BUY" if side == SIDE_BUY else "SELL"
            current_price = self.data_fetcher.get_current_price(symbol)
            if current_price is None or current_price <= 0:
                raise ValueError(f"Invalid price for {symbol}: {current_price}")
            trade_value = current_price * quantity
            if action == "BUY":
                if trade_value > self.current_balance:
                    raise ValueError(f"Insufficient balance: {self.current_balance:.2f} USDT, needed {trade_value:.2f} USDT")
                self.current_balance -= trade_value
                self.in_position = True
                self.buy_price = current_price  # Update buy price only on successful buy
                self.chosen_symbol = symbol
            else:  # SELL
                self.current_balance += trade_value
                self.in_position = False
                self.buy_price = 0.0  # Reset buy price after sell
                self.chosen_symbol = None
            log_trade(symbol, action, current_price, quantity, profit=None if action == "BUY" else (trade_value - (self.buy_price * quantity if self.buy_price > 0 else 0)))
            logging.info(f"Order placed: {action} {quantity} {symbol} at {current_price:.6f}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
            print(f"Order placed: {action} {quantity} {symbol} at {current_price:.6f}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
            return True
        except Exception as e:
            logging.error(f"Error placing order for {symbol}: {e}")
            current_price = self.data_fetcher.get_current_price(symbol) or 0.0
            log_trade(symbol, action, current_price, quantity, error=str(e))
            print(f"Error placing order for {symbol}: {e}")
            return False

    def calculate_dynamic_threshold(self, indicators, strategy='momentum'):
        """Calculate a dynamic prediction threshold based on ATR, RSI, and strategy (from paper on March 1, 2025)."""
        atr = indicators['atr'] if indicators['atr'] > 0 else 0.0001  # Avoid zero
        rsi = indicators['rsi']
        base_threshold = PRED_THRESHOLD  # From config, 0.4 for more opportunities
        if strategy == 'momentum':
            volatility_factor = max(0.3, min(0.9, 1 - (atr * 1000)))  # Normalize ATR
            momentum_factor = 0.5 if rsi < 30 else (0.6 if rsi < 50 else 0.8)  # Lower for oversold, higher for neutral
        elif strategy == 'mean_reversion':
            volatility_factor = max(0.4, min(1.0, atr * 1000))  # Higher for volatile reversals
            momentum_factor = 0.7 if rsi > 60 or rsi < 40 else 0.5  # Relaxed RSI thresholds
        else:  # volatility
            volatility_factor = max(0.2, min(0.8, 1 / (atr * 1000)))  # Lower for high volatility
            momentum_factor = 0.6
        return base_threshold * volatility_factor * momentum_factor

    def calculate_dynamic_limits(self, atr, strategy='momentum'):
        """Calculate dynamic stop-loss and take-profit based on ATR and strategy (from paper on March 1, 2025)."""
        atr = atr if atr > 0 else 0.0001
        volatility_factor = max(0.5, min(1.5, atr * 1000))  # Normalize ATR
        if strategy == 'momentum':
            dynamic_stop_loss = STOP_LOSS_BASE * volatility_factor * 1.2  # Slightly tighter for trends
            dynamic_take_profit = TAKE_PROFIT_BASE * volatility_factor * 1.5  # Higher for trends
        elif strategy == 'mean_reversion':
            dynamic_stop_loss = STOP_LOSS_BASE * volatility_factor * 1.5  # Wider for reversals
            dynamic_take_profit = TAKE_PROFIT_BASE * volatility_factor * 1.2  # Lower for quick reversals
        else:  # volatility
            dynamic_stop_loss = STOP_LOSS_BASE * volatility_factor * 1.0  # Standard for volatility
            dynamic_take_profit = TAKE_PROFIT_BASE * volatility_factor * 1.3  # Slightly higher for breakouts
        return dynamic_stop_loss, dynamic_take_profit

    def calculate_multi_timeframe_indicators(self, symbol):
        """Calculate technical indicators for multiple timeframes (1m, 5m, 15m)."""
        multi_data = self.data_fetcher.get_multi_timeframe_data(symbol)
        indicators = {}
        for interval, df in multi_data.items():
            if df is not None and not df.empty:
                indicators[f'sma_short_{interval}'] = df['SMA_Short'].iloc[-1] if not pd.isna(df['SMA_Short'].iloc[-1]) and df['SMA_Short'].iloc[-1] > 0 else 0.0001
                indicators[f'sma_long_{interval}'] = df['SMA_Long'].iloc[-1] if not pd.isna(df['SMA_Long'].iloc[-1]) and df['SMA_Long'].iloc[-1] > 0 else 0.0001
                indicators[f'rsi_' + interval] = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] > 0 else 50.0
                indicators[f'atr_' + interval] = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) and df['ATR'].iloc[-1] > 0 else 0.0001
                indicators[f'price_' + interval] = df['close'].iloc[-1] if not pd.isna(df['close'].iloc[-1]) and df['close'].iloc[-1] > 0 else 0.0001
                indicators[f'upper_band_' + interval] = df['Upper_Band'].iloc[-1] if not pd.isna(df['Upper_Band'].iloc[-1]) and df['Upper_Band'].iloc[-1] > 0 else 0.0001
                indicators[f'lower_band_' + interval] = df['Lower_Band'].iloc[-1] if not pd.isna(df['Lower_Band'].iloc[-1]) and df['Lower_Band'].iloc[-1] > 0 else 0.0001
        return indicators if all(v > 0 for v in indicators.values() if v != 50.0) else None

    def calculate_indicators(self, symbol, strategy='momentum'):
        """Calculate technical indicators for trading decisions, using historical data, with fallback for invalid data and mean reversion fixes."""
        df = self.data_fetcher.get_historical_data(symbol, limit=SMA_LONG_PERIOD + 10, use_cache=False)  # Force fresh data
        if df is not None and not df.empty:
            indicators = {
                'sma_short': df['SMA_Short'].iloc[-1] if not pd.isna(df['SMA_Short'].iloc[-1]) and df['SMA_Short'].iloc[-1] > 0 else 0.0001,
                'sma_long': df['SMA_Long'].iloc[-1] if not pd.isna(df['SMA_Long'].iloc[-1]) and df['SMA_Long'].iloc[-1] > 0 else 0.0001,
                'rsi': df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] > 0 else 50.0,
                'atr': df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) and df['ATR'].iloc[-1] > 0 else 0.0001,
                'price': df['close'].iloc[-1] if not pd.isna(df['close'].iloc[-1]) and df['close'].iloc[-1] > 0 else 0.0001,  # Added 'price' explicitly
                'upper_band': df['Upper_Band'].iloc[-1] if not pd.isna(df['Upper_Band'].iloc[-1]) and df['Upper_Band'].iloc[-1] > 0 else 0.0001,
                'lower_band': df['Lower_Band'].iloc[-1] if not pd.isna(df['Lower_Band'].iloc[-1]) and df['Lower_Band'].iloc[-1] > 0 else 0.0001
            }
            if strategy == 'mean_reversion':
                # Ensure Z-score calculation handles partial data and NaNs
                sma_long_series = df['SMA_Long'].dropna()
                if len(sma_long_series) > SMA_LONG_PERIOD and np.std(sma_long_series) > 0:
                    indicators['z_score'] = (indicators['price'] - indicators['sma_long']) / np.std(sma_long_series)
                else:
                    indicators['z_score'] = 0.0  # Default to 0 if insufficient data
            return indicators if all(v > 0 for v in indicators.values() if v != 50.0) else None
        logging.warning(f"No valid indicators for {symbol}, attempting fresh fetch failed - check logs/data_fetch.log")
        return None

    def get_prediction(self, symbol):
        """Get LSTM prediction for price movement, using historical data, with fallback for invalid data."""
        df = self.data_fetcher.get_historical_data(symbol, limit=LOOKBACK, use_cache=False)  # Force fresh data
        if df is not None and not df.empty:
            prices = df['close'].values[-LOOKBACK:] if not pd.isna(df['close'].values[-LOOKBACK:]).any() and np.all(df['close'].values[-LOOKBACK:] > 0) else np.ones(LOOKBACK) * 0.0001  # Default to small positive value
            volumes = df['volume'].values[-LOOKBACK:] if not pd.isna(df['volume'].values[-LOOKBACK:]).any() and np.all(df['volume'].values[-LOOKBACK:] > 0) else np.ones(LOOKBACK) * 0.0001
            rsi = df['RSI'].values[-LOOKBACK:] if not pd.isna(df['RSI'].values[-LOOKBACK:]).any() and np.all(df['RSI'].values[-LOOKBACK:] > 0) else np.full(LOOKBACK, 50.0)
            macd = df['MACD'].values[-LOOKBACK:] if not pd.isna(df['MACD'].values[-LOOKBACK:]).any() and np.all(df['MACD'].values[-LOOKBACK:] > 0) else np.zeros(LOOKBACK)
            upper_band = df['Upper_Band'].values[-LOOKBACK:] if not pd.isna(df['Upper_Band'].values[-LOOKBACK:]).any() and np.all(df['Upper_Band'].values[-LOOKBACK:] > 0) else np.ones(LOOKBACK) * 0.0001
            lower_band = df['Lower_Band'].values[-LOOKBACK:] if not pd.isna(df['Lower_Band'].values[-LOOKBACK:]).any() and np.all(df['Lower_Band'].values[-LOOKBACK:] > 0) else np.ones(LOOKBACK) * 0.0001
            atr = df['ATR'].values[-LOOKBACK:] if not pd.isna(df['ATR'].values[-LOOKBACK:]).any() and np.all(df['ATR'].values[-LOOKBACK:] > 0) else np.ones(LOOKBACK) * 0.0001
            sma_short = df['SMA_Short'].values[-LOOKBACK:] if not pd.isna(df['SMA_Short'].values[-LOOKBACK:]).any() and np.all(df['SMA_Short'].values[-LOOKBACK:] > 0) else np.ones(LOOKBACK) * 0.0001
            sma_long = df['SMA_Long'].values[-LOOKBACK:] if not pd.isna(df['SMA_Long'].values[-LOOKBACK:]).any() and np.all(df['SMA_Long'].values[-LOOKBACK:] > 0) else np.ones(LOOKBACK) * 0.0001
            X = self.preprocessor.prepare_prediction_data(df)
            try:
                prediction = self.model.predict(X, verbose=0)[0][0]
                return max(0.0, min(1.0, prediction))  # Ensure prediction is between 0 and 1
            except Exception as e:
                logging.error(f"Error predicting for {symbol}: {e}")
                return 0.5  # Default to neutral prediction if error occurs
        logging.warning(f"No valid data for prediction on {symbol}, attempting fresh fetch failed - check logs/data_fetch.log")
        return 0.5  # Default to neutral prediction if no valid data

    def calculate_profit(self, current_price):
        """Calculate profit/loss percentage based on current price and buy price."""
        if self.buy_price <= 0 or current_price <= 0:
            return 0.0  # Return 0 if no valid buy price or current price
        profit_pct = (current_price - self.buy_price) / self.buy_price * 100
        return profit_pct

    def run(self):
        """Run the trading bot to maximize profits while minimizing losses, using historical data, with robust error handling and advanced strategies from the paper on March 1, 2025."""
        print(f"🔥 Starting Trading Bot: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR) to {(self.current_balance * (1 + TAKE_PROFIT_BASE)):.2f} USDT in {RUNTIME//60} minutes")
        start_time = time.time()
        last_update = start_time

        while time.time() - start_time < RUNTIME:
            current_time = time.time()
            if current_time - last_update >= 1:  # Update every second
                print(f"Time remaining: {RUNTIME - (current_time - start_time):.1f} seconds, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                try:
                    if not self.in_position:
                        print("Evaluating symbols for trades, prioritizing low-priced pairs...")
                        sorted_symbols = sorted(SYMBOLS, key=lambda x: self.data_fetcher.get_current_price(x) or float('inf'))
                        for symbol in sorted_symbols:
                            print(f"Processing {symbol}...")
                            # Test momentum strategy first (trending markets like SOLUSDT)
                            momentum_indicators = self.calculate_indicators(symbol, strategy='momentum')
                            if momentum_indicators is None:
                                print(f"⚠️ Skipping {symbol} (momentum) due to no valid indicators, attempting fresh fetch")
                                continue
                            multi_indicators = self.calculate_multi_timeframe_indicators(symbol)
                            if multi_indicators is None:
                                print(f"⚠️ Skipping {symbol} due to no valid multi-timeframe indicators")
                                continue
                            prediction = self.get_prediction(symbol)
                            if pd.isna(prediction) or prediction <= 0:
                                logging.warning(f"Invalid prediction for {symbol}: {prediction}")
                                continue
                            sentiment = self.data_fetcher.get_sentiment(symbol)
                            sentiment_weight = 0.4 if abs(sentiment) > 0.5 else 0.2  # Increase weight for strong sentiment
                            combined_score = 0.6 * prediction + sentiment_weight * sentiment  # Adjust weights for balance

                            df = self.data_fetcher.get_historical_data(symbol, limit=LOOKBACK, use_cache=False)  # Force fresh data
                            if df is not None and not df.empty:
                                macd = df['MACD'].iloc[-1]
                                signal = df['Signal_Line'].iloc[-1]
                                if macd > signal and macd > 0:  # Bullish MACD crossover (momentum)
                                    combined_score += 0.1  # Boost score for bullish momentum

                            current_price = momentum_indicators['price']
                            print(f"📈 {symbol} (Momentum): Price {current_price:.6f}, LSTM {prediction:.2f}, Sentiment {sentiment:.2f}, Combined {combined_score:.2f}, RSI {momentum_indicators['rsi']:.2f}, SMA_Short {momentum_indicators['sma_short']:.2f}, SMA_Long {momentum_indicators['sma_long']:.2f}, ATR {momentum_indicators['atr']:.6f}")

                            dynamic_threshold = self.calculate_dynamic_threshold(momentum_indicators, strategy='momentum')
                            if (combined_score > dynamic_threshold and 
                                all(multi_indicators[f'sma_short_{interval}'] > multi_indicators[f'sma_long_{interval}'] for interval in ['1m', '5m', '15m']) and 
                                any(multi_indicators[f'rsi_' + interval] < 50 for interval in ['1m', '5m', '15m']) and 
                                momentum_indicators['price'] > momentum_indicators['upper_band'] and  # Momentum breakout
                                current_price > 0 and 
                                not pd.isna(current_price)):
                                quantity = self.adjust_quantity(symbol, current_price, momentum_indicators['atr'])
                                if quantity > 0 and self.current_balance >= (current_price * quantity):
                                    print(f"Attempting to buy {quantity} {symbol} at {current_price:.6f} (Momentum)")
                                    if self.place_order(symbol, SIDE_BUY, quantity):
                                        print(f"Bought {symbol} at {current_price:.6f}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                                        self.strategy = 'momentum'
                                        break  # Exit after first successful buy
                                else:
                                    print(f"⚠️ Insufficient balance or invalid quantity for {symbol}, skipping buy. Balance: {self.current_balance:.2f} USDT")

                            # Test mean reversion strategy (stable markets like DOGEUSDT)
                            mean_reversion_indicators = self.calculate_indicators(symbol, strategy='mean_reversion')
                            if mean_reversion_indicators is None:
                                print(f"⚠️ Skipping {symbol} (mean reversion) due to no valid indicators, attempting fresh fetch")
                                continue
                            z_score = mean_reversion_indicators['z_score']
                            if abs(z_score) > 0.8 and (z_score < -0.8 or z_score > 0.8):  # Relaxed from 1.0 for more opportunities
                                combined_score = 0.8 * prediction + 0.2 * sentiment  # Reset for mean reversion
                                dynamic_threshold = self.calculate_dynamic_threshold(mean_reversion_indicators, strategy='mean_reversion')
                                if (combined_score > dynamic_threshold and 
                                    any(multi_indicators[f'rsi_' + interval] > 60 or multi_indicators[f'rsi_' + interval] < 40 for interval in ['1m', '5m']) and  # Relaxed RSI thresholds
                                    current_price > 0 and 
                                    not pd.isna(current_price)):
                                    quantity = self.adjust_quantity(symbol, current_price, mean_reversion_indicators['atr'])
                                    if quantity > 0 and self.current_balance >= (current_price * quantity):
                                        print(f"Attempting to buy {quantity} {symbol} at {current_price:.6f} (Mean Reversion)")
                                        if self.place_order(symbol, SIDE_BUY, quantity):
                                            print(f"Bought {symbol} at {current_price:.6f}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                                            self.strategy = 'mean_reversion'
                                            break  # Exit after first successful buy
                                    else:
                                        print(f"⚠️ Insufficient balance or invalid quantity for {symbol}, skipping buy. Balance: {self.current_balance:.2f} USDT")

                            # Test volatility strategy (high-volatility markets like SOLUSDT)
                            volatility_indicators = self.calculate_indicators(symbol, strategy='volatility')
                            if volatility_indicators is None:
                                print(f"⚠️ Skipping {symbol} (volatility) due to no valid indicators")
                                continue
                            atr = volatility_indicators['atr']
                            if atr > 0.001:  # Lowered from 0.005 for more opportunities on March 1, 2025
                                combined_score = 0.7 * prediction + 0.3 * sentiment  # Higher sentiment weight for volatility
                                dynamic_threshold = self.calculate_dynamic_threshold(volatility_indicators, strategy='volatility')
                                if (combined_score > dynamic_threshold and 
                                    volatility_indicators['price'] > volatility_indicators['upper_band'] and  # Volatility breakout
                                    current_price > 0 and 
                                    not pd.isna(current_price)):
                                    quantity = self.adjust_quantity(symbol, current_price, volatility_indicators['atr'])
                                    if quantity > 0 and self.current_balance >= (current_price * quantity):
                                        print(f"Attempting to buy {quantity} {symbol} at {current_price:.6f} (Volatility)")
                                        if self.place_order(symbol, SIDE_BUY, quantity):
                                            print(f"Bought {symbol} at {current_price:.6f}, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                                            self.strategy = 'volatility'
                                            break  # Exit after first successful buy
                                    else:
                                        print(f"⚠️ Insufficient balance or invalid quantity for {symbol}, skipping buy. Balance: {self.current_balance:.2f} USDT")

                    else:
                        print(f"Holding {self.chosen_symbol}...")
                        current_price = self.data_fetcher.get_current_price(self.chosen_symbol)
                        if current_price is not None and current_price > 0:
                            profit_pct = self.calculate_profit(current_price)
                            indicators = self.calculate_indicators(self.chosen_symbol, strategy=self.strategy)  # Use current strategy
                            if indicators is None:
                                print(f"⚠️ Skipping position update for {self.chosen_symbol} due to no valid indicators")
                                continue
                            atr = indicators['atr'] if not pd.isna(indicators['atr']) else 0.0001
                            quantity = self.adjust_quantity(self.chosen_symbol, current_price, atr)
                            print(f"📊 {self.chosen_symbol}: Price {current_price:.6f}, Profit {profit_pct:.2f}%, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR), ATR {atr:.6f}")
                            dynamic_stop_loss, dynamic_take_profit = self.calculate_dynamic_limits(atr, strategy=self.strategy)
                            if profit_pct >= dynamic_take_profit * 100:
                                print(f"Attempting to sell {quantity} {self.chosen_symbol} at {current_price:.6f} for profit")
                                if self.place_order(self.chosen_symbol, SIDE_SELL, quantity):
                                    print(f"🎉 Profit: {profit_pct:.2f}%, New Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                                    break
                            elif profit_pct <= dynamic_stop_loss * 100 or (self.buy_price > 0 and current_price > 0 and (current_price - self.buy_price) / self.buy_price <= -atr * 0.5):
                                print(f"Attempting to sell {quantity} {self.chosen_symbol} at {current_price:.6f} for loss")
                                if self.place_order(self.chosen_symbol, SIDE_SELL, quantity):
                                    print(f"❌ Loss: {profit_pct:.2f}%, New Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                                    break
                        else:
                            print(f"⚠️ Skipping position update for {self.chosen_symbol} due to invalid price. Retrying...")
                            time.sleep(2)
                            current_price = self.data_fetcher.get_current_price(self.chosen_symbol)
                            if current_price is not None and current_price > 0:
                                profit_pct = self.calculate_profit(current_price)
                                indicators = self.calculate_indicators(self.chosen_symbol, strategy=self.strategy)
                                if indicators is None:
                                    print(f"⚠️ Skipping retry for {self.chosen_symbol} due to no valid indicators")
                                    continue
                                atr = indicators['atr'] if not pd.isna(indicators['atr']) else 0.0001
                                quantity = self.adjust_quantity(self.chosen_symbol, current_price, atr)
                                print(f"📊 {self.chosen_symbol} (retry): Price {current_price:.6f}, Profit {profit_pct:.2f}%, Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR), ATR {atr:.6f}")
                                dynamic_stop_loss, dynamic_take_profit = self.calculate_dynamic_limits(atr, strategy=self.strategy)
                                if profit_pct >= dynamic_take_profit * 100:
                                    print(f"Attempting to sell {quantity} {self.chosen_symbol} at {current_price:.6f} for profit")
                                    if self.place_order(self.chosen_symbol, SIDE_SELL, quantity):
                                        print(f"🎉 Profit: {profit_pct:.2f}%, New Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                                        break
                                elif profit_pct <= dynamic_stop_loss * 100 or (self.buy_price > 0 and current_price > 0 and (current_price - self.buy_price) / self.buy_price <= -atr * 0.5):
                                    print(f"Attempting to sell {quantity} {self.chosen_symbol} at {current_price:.6f} for loss")
                                    if self.place_order(self.chosen_symbol, SIDE_SELL, quantity):
                                        print(f"❌ Loss: {profit_pct:.2f}%, New Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                                        break
                except Exception as e:
                    logging.error(f"Error in trading loop: {e}")
                    print(f"Error in trading loop: {e}")
                    time.sleep(5)
                last_update = current_time

        if self.in_position:
            print(f"Time's up, checking final position for {self.chosen_symbol}...")
            current_price = self.data_fetcher.get_current_price(self.chosen_symbol)
            if current_price is not None and current_price > 0:
                quantity = self.adjust_quantity(self.chosen_symbol, current_price)
                if self.place_order(self.chosen_symbol, SIDE_SELL, quantity):
                    profit_pct = self.calculate_profit(current_price)
                    print(f"⏰ Time’s up! Final Result: {profit_pct:.2f}%, Final Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
            else:
                print(f"⚠️ Failed to get final price for {self.chosen_symbol}, cannot sell. Retrying...")
                time.sleep(2)
                current_price = self.data_fetcher.get_current_price(self.chosen_symbol)
                if current_price is not None and current_price > 0:
                    quantity = self.adjust_quantity(self.chosen_symbol, current_price)
                    if self.place_order(self.chosen_symbol, SIDE_SELL, quantity):
                        profit_pct = self.calculate_profit(current_price)
                        print(f"⏰ Retry succeeded! Final Result: {profit_pct:.2f}%, Final Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")
                else:
                    print(f"⚠️ Final retry failed for {self.chosen_symbol}, position not closed, Final Balance: {self.current_balance:.2f} USDT (~{(self.current_balance / INR_TO_USDT_RATE):.2f} INR)")

    def get_account_balance(self):
        """Fetch current USDT balance from Binance Testnet (simplified)."""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free']) if balance['free'] and float(balance['free']) > 0 else self.current_balance
            return self.current_balance  # Fallback to tracked balance
        except Exception as e:
            logging.error(f"Error fetching balance: {e}")
            return self.current_balance

    def get_precision(self, symbol):
        """Get precision for a symbol's price."""
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', symbol_info['filters']), None)
            return int(round(-np.log10(float(price_filter['tickSize']))))
        except Exception as e:
            logging.error(f"Error getting precision for {symbol}: {e}")
            return 6  # Default precision

    def calculate_var(self, returns, confidence=0.95):
        """Calculate Value at Risk."""
        if len(returns) < 2 or np.all(returns == 0):
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.01):
        """Calculate Sharpe Ratio for performance evaluation."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) - risk_free_rate) / np.std(returns)