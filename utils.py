# utils.py
import logging
import time
import numpy as np
import pandas as pd  # Added pandas import
from config import PRED_THRESHOLD, TAKE_PROFIT_BASE, STOP_LOSS_BASE, TRADING_LOG, BALANCE, RISK_FREE_RATE, LOOKBACK, SMA_LONG_PERIOD

def setup_logging(log_file):
    """Configure logging to file."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    return logging.getLogger()

def log_trade(symbol, action, price, quantity, profit=None, error=None):
    logger = setup_logging(TRADING_LOG)
    message = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} - {action} at {price:.6f}, Quantity: {quantity:.6f}, Balance: {BALANCE:.2f} USDT (~{(BALANCE / 0.012):.2f} INR)"
    if profit is not None:
        message += f", Profit: {profit:.2f} USDT (~{(profit / 0.012):.2f} INR)"
    if error:
        message += f", Error: {error}"
    logger.info(message)
    print(message)  # Print to console for immediate feedback

def backtest_strategy(data_fetcher, model, preprocessor, symbols, lookback=LOOKBACK):
    """Backtest the trading strategies (momentum, mean reversion, volatility) on historical data with risk management, using fresh historical data, with robust error handling and scaler fitting on March 1, 2025."""
    profits = []
    total_trades = 0
    balance = BALANCE
    print(f"Starting backtest with symbols: {symbols} and balance: {balance:.2f} USDT")
    strategies = ['momentum', 'mean_reversion', 'volatility']
    for symbol in symbols:
        print(f"Backtesting {symbol} with strategies: {strategies}...")
        df = data_fetcher.get_historical_data(symbol, limit=2000, use_cache=False)  # Force fresh data
        symbol_trades = {strategy: 0 for strategy in strategies}
        position = {strategy: 0 for strategy in strategies}
        buy_price = {strategy: 0.0 for strategy in strategies}
        if df is not None and not df.empty and not df['close'].isna().all() and not df['close'].eq(0).all():
            print(f"Fetched {len(df)} candles for {symbol} from historical data")
            prices = df['close'].values
            volumes = df['volume'].values
            rsi = df['RSI'].values
            macd = df['MACD'].values
            upper_band = df['Upper_Band'].values
            lower_band = df['Lower_Band'].values
            atr = df['ATR'].values
            sma_short = df['SMA_Short'].values
            sma_long = df['SMA_Long'].values
            
            # Fit the scaler with the first chunk of valid data to ensure it's fitted
            sample_df = df.iloc[:min(len(df), lookback * 2)]  # Use initial data for fitting
            sample_prices = sample_df['close'].values.reshape(-1, 1)
            sample_volumes = sample_df['volume'].values.reshape(-1, 1)
            sample_rsi = sample_df['RSI'].values.reshape(-1, 1)
            sample_macd = sample_df['MACD'].values.reshape(-1, 1)
            sample_upper_band = sample_df['Upper_Band'].values.reshape(-1, 1)
            sample_lower_band = sample_df['Lower_Band'].values.reshape(-1, 1)
            sample_atr = sample_df['ATR'].values.reshape(-1, 1)
            sample_sma_short = sample_df['SMA_Short'].values.reshape(-1, 1)
            sample_sma_long = sample_df['SMA_Long'].values.reshape(-1, 1)
            features = np.column_stack((sample_prices, sample_volumes, sample_rsi, sample_macd, sample_upper_band, sample_lower_band, sample_atr, sample_sma_short, sample_sma_long))
            if not preprocessor.is_fitted and features.size > 0 and not np.any(np.isnan(features)) and not np.any(features == 0):
                preprocessor.fit_scaler(features)
                print("Fitted MinMaxScaler for backtesting")

            for i in range(lookback, len(prices) - 5):
                try:
                    X = preprocessor.prepare_prediction_data(df.iloc[i - lookback:i])
                    # print(f"Predicting for {symbol} at index {i}, X shape: {X.shape}")
                    prediction = model.predict(X, verbose=0)[0][0]
                    current_price = prices[i]
                    indicators = {
                        'sma_short': sma_short[i] if not pd.isna(sma_short[i]) and sma_short[i] > 0 else 0.0001,
                        'sma_long': sma_long[i] if not pd.isna(sma_long[i]) and sma_long[i] > 0 else 0.0001,
                        'rsi': rsi[i] if not pd.isna(rsi[i]) and rsi[i] > 0 else 50.0,
                        'atr': atr[i] if not pd.isna(atr[i]) and atr[i] > 0 else 0.0001,
                        'price': current_price if not pd.isna(current_price) and current_price > 0 else 0.0001,  # Added 'price' key
                        'upper_band': upper_band[i] if not pd.isna(upper_band[i]) and upper_band[i] > 0 else 0.0001,
                        'lower_band': lower_band[i] if not pd.isna(lower_band[i]) and lower_band[i] > 0 else 0.0001
                    }
                    if any(pd.isna(v) or v <= 0 for v in [indicators['price'], indicators['sma_short'], indicators['sma_long'], indicators['rsi'], indicators['atr'], indicators['upper_band'], indicators['lower_band']]):
                        logging.warning(f"Invalid or zero data at index {i} for {symbol}")
                        continue

                    # Momentum Strategy (trending markets, e.g., SOLUSDT at $140.960000, RSI 57.66)
                    momentum_indicators = indicators.copy()
                    momentum_threshold = PRED_THRESHOLD  # Use base for simplicity, or dynamic from trade.py
                    if (prediction > momentum_threshold and 
                        momentum_indicators['sma_short'] > momentum_indicators['sma_long'] and 
                        momentum_indicators['rsi'] < 50 and 
                        momentum_indicators['price'] > momentum_indicators['upper_band'] and  # Breakout for momentum
                        indicators['price'] > 0 and 
                        not pd.isna(indicators['price']) and 
                        not position['momentum']):
                        quantity = min((balance * (0.5 + 0.4 * max(0.2, min(1.0, 1 / (momentum_indicators['atr'] * 1000))))) / indicators['price'], 1000000)  # Use indicators['price']
                        step_size = data_fetcher.get_lot_size(symbol)
                        if step_size and step_size > 0:
                            quantity = round(quantity / step_size) * step_size
                        if quantity > 0 and balance >= (indicators['price'] * quantity) and indicators['price'] > 0:
                            symbol_trades['momentum'] += 1
                            total_trades += 1
                            position['momentum'] = quantity
                            buy_price['momentum'] = indicators['price']
                            balance -= indicators['price'] * quantity
                            print(f"Backtest trade for {symbol} #{symbol_trades['momentum']} (Momentum): Buy at {buy_price['momentum']:.6f}, Balance: {balance:.2f} USDT")

                    # Mean Reversion Strategy (stable markets, e.g., DOGEUSDT at $0.204020, RSI 52.88)
                    mean_reversion_indicators = indicators.copy()
                    mean_reversion_indicators['z_score'] = (indicators['price'] - mean_reversion_indicators['sma_long']) / df['SMA_Long'].std() if len(df) > SMA_LONG_PERIOD else 0.0
                    if not position['mean_reversion'] and abs(mean_reversion_indicators['z_score']) > 1.0 and (mean_reversion_indicators['z_score'] < -1.0 or mean_reversion_indicators['z_score'] > 1.0):
                        mean_reversion_threshold = PRED_THRESHOLD * 0.7  # Lower threshold for mean reversion
                        if (prediction > mean_reversion_threshold and 
                            (mean_reversion_indicators['rsi'] > 70 or mean_reversion_indicators['rsi'] < 30) and  # Overbought/oversold
                            indicators['price'] > 0 and 
                            not pd.isna(indicators['price'])):
                            quantity = min((balance * (0.5 + 0.4 * max(0.2, min(1.0, 1 / (mean_reversion_indicators['atr'] * 1000))))) / indicators['price'], 1000000)
                            step_size = data_fetcher.get_lot_size(symbol)
                            if step_size and step_size > 0:
                                quantity = round(quantity / step_size) * step_size
                            if quantity > 0 and balance >= (indicators['price'] * quantity) and indicators['price'] > 0:
                                symbol_trades['mean_reversion'] += 1
                                total_trades += 1
                                position['mean_reversion'] = quantity
                                buy_price['mean_reversion'] = indicators['price']
                                balance -= indicators['price'] * quantity
                                print(f"Backtest trade for {symbol} #{symbol_trades['mean_reversion']} (Mean Reversion): Buy at {buy_price['mean_reversion']:.6f}, Balance: {balance:.2f} USDT")

                    # Volatility Strategy (high-volatility markets, e.g., SOLUSDT ATR > 0.005)
                    volatility_indicators = indicators.copy()
                    if not position['volatility'] and volatility_indicators['atr'] > 0.005:  # High volatility threshold on March 1, 2025
                        volatility_threshold = PRED_THRESHOLD * 0.6  # Lower threshold for volatility
                        if (prediction > volatility_threshold and 
                            volatility_indicators['price'] > volatility_indicators['upper_band'] and  # Volatility breakout
                            indicators['price'] > 0 and 
                            not pd.isna(indicators['price'])):
                            quantity = min((balance * (0.5 + 0.4 * max(0.2, min(1.0, 1 / (volatility_indicators['atr'] * 1000))))) / indicators['price'], 1000000)
                            step_size = data_fetcher.get_lot_size(symbol)
                            if step_size and step_size > 0:
                                quantity = round(quantity / step_size) * step_size
                            if quantity > 0 and balance >= (indicators['price'] * quantity) and indicators['price'] > 0:
                                symbol_trades['volatility'] += 1
                                total_trades += 1
                                position['volatility'] = quantity
                                buy_price['volatility'] = indicators['price']
                                balance -= indicators['price'] * quantity
                                print(f"Backtest trade for {symbol} #{symbol_trades['volatility']} (Volatility): Buy at {buy_price['volatility']:.6f}, Balance: {balance:.2f} USDT")

                    # Sell logic for all strategies
                    for strategy in strategies:
                        if position[strategy] and buy_price[strategy] > 0 and indicators['price'] > 0:
                            profit_pct = (indicators['price'] - buy_price[strategy]) / buy_price[strategy] * 100 if buy_price[strategy] > 0 else 0.0
                            dynamic_stop_loss, dynamic_take_profit = self.calculate_dynamic_limits(indicators['atr'], strategy)
                            if (profit_pct >= dynamic_take_profit * 100 or 
                                profit_pct <= dynamic_stop_loss * 100 or 
                                (indicators['price'] - buy_price[strategy]) / buy_price[strategy] <= -indicators['atr'] * 0.5 if indicators['atr'] > 0 else False):
                                balance += position[strategy] * indicators['price']
                                print(f"Backtest trade for {symbol} #{symbol_trades[strategy]} ({strategy.capitalize()}): Sell at {indicators['price']:.6f}, Profit {profit_pct:.2f}%, Balance: {balance:.2f} USDT")
                                if profit_pct >= dynamic_take_profit * 100:
                                    profits.append(profit_pct)
                                elif profit_pct <= dynamic_stop_loss * 100:
                                    profits.append(profit_pct)
                                position[strategy] = 0
                                buy_price[strategy] = 0.0

                except Exception as e:
                    logging.error(f"Error in backtest for {symbol} at index {i}: {e}")
                    print(f"Error in backtest for {symbol} at index {i}: {e}")
                    continue

            for strategy in strategies:
                print(f"Completed backtest for {symbol} with {symbol_trades[strategy]} {strategy.capitalize()} trades")
        else:
            logging.error(f"Failed to fetch valid data for {symbol} - check logs/data_fetch.log")
            print(f"Failed to fetch valid data for {symbol}, skipping...")
            continue

    avg_profit = sum(profits) / len(profits) if profits else 0.0
    sharpe = calculate_sharpe_ratio(profits, RISK_FREE_RATE) if profits else 0.0
    var = calculate_var(profits) if profits else 0.0
    print(f"Backtest completed. Total trades: {total_trades}, Average Profit: {avg_profit:.2f}%, Sharpe Ratio: {sharpe:.2f}, VaR (95%): {var:.2f}% with {balance:.2f} USDT")
    return avg_profit

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """Calculate Sharpe Ratio for performance evaluation."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk."""
    if len(returns) < 2 or np.all(returns == 0):
        return 0.0
    return np.percentile(returns, (1 - confidence) * 100)