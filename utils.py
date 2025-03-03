# utils.py
import logging
import time
import numpy as np
import pandas as pd
from config import RSI_PERIOD,PRED_THRESHOLD, TAKE_PROFIT_BASE, STOP_LOSS_BASE, TRADING_LOG, INR_TO_USDT_RATE, RISK_FREE_RATE, LOOKBACK, SMA_FAST_PERIOD, SMA_SLOW_PERIOD, RSI_PERIOD, ATR_PERIOD, BOLINGER_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL

def setup_logging(log_file):
    """Configure logging to file."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    return logging.getLogger()

def log_trade(symbol, action, price, quantity, balance=None, profit=None, error=None):
    """Log a trade with dynamic balance and INR conversion."""
    balance = balance if balance is not None else 0.0  # Default to 0 if balance not provided
    logger = setup_logging(TRADING_LOG)
    message = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} - {action} at {price:.6f}, Quantity: {quantity:.6f}, Balance: {balance:.2f} USDT (~{(balance / INR_TO_USDT_RATE):.2f} INR)"
    if profit is not None:
        message += f", Profit: {profit:.2f} USDT (~{(profit / INR_TO_USDT_RATE):.2f} INR)"
    if error:
        message += f", Error: {error}"
    logger.info(message)
    print(message)  # Print to console for immediate feedback

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate Sharpe Ratio for performance evaluation."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk."""
    if len(returns) < 2 or np.all(returns == 0):
        return 0.0
    return np.percentile(returns, (1 - confidence) * 100)

class Backtest:
    def __init__(self, trading_bot, initial_balance=None):
        """Initialize Backtest with a TradingBot instance and optional initial balance."""
        self.trading_bot = trading_bot
        self.initial_balance = initial_balance if initial_balance is not None else trading_bot.current_balance  # Use trading_bot’s balance

    def backtest_strategy(self, data_fetcher, model, preprocessor, symbols, lookback=LOOKBACK):
        """Backtest the trading strategies with enhanced metrics and tracking."""
        profits = []
        total_trades = 0
        balance = self.initial_balance  # Use dynamic initial balance
        trades_log = []  # Track trade details for performance analysis
        logger = setup_logging(TRADING_LOG)
        logger.info(f"Starting backtest with symbols: {symbols} and balance: {balance:.2f} USDT")
        strategies = ['momentum', 'mean_reversion', 'volatility']
        
        for symbol in symbols:
            logger.info(f"Backtesting {symbol} with strategies: {strategies}...")
            df = data_fetcher.get_historical_data(symbol, limit=max(SMA_SLOW_PERIOD + 10, LOOKBACK), use_cache=False)
            symbol_trades = {strategy: 0 for strategy in strategies}
            position = {strategy: 0 for strategy in strategies}
            buy_price = {strategy: 0.0 for strategy in strategies}
            if df is not None and not df.empty and not df['close'].isna().all() and not df['close'].eq(0).all():
                logger.info(f"Fetched {len(df)} candles for {symbol} from historical data")
                prices = df['close'].values
                volumes = df['volume'].values
                rsi = df['RSI'].values
                macd = df['MACD'].values
                upper_band = df['Upper_Band'].values
                lower_band = df['Lower_Band'].values
                atr = df['ATR'].values
                sma_fast = df['SMA_Fast'].values
                sma_slow = df['SMA_Slow'].values
                
                # Fit the scaler with the first chunk of valid data to ensure it's fitted
                sample_df = df.iloc[:min(len(df), lookback * 2)]
                sample_features = ['close', 'volume', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band', 
                                  'ATR', 'SMA_Fast', 'SMA_Slow', 'bid_price', 'ask_price', 'spread', 'funding_rate']
                sample_data = sample_df[sample_features].fillna(0).replace([np.inf, -np.inf], 0).values
                if not preprocessor.is_fitted and sample_data.size > 0 and not np.any(np.isnan(sample_data)) and not np.any(sample_data == 0):
                    preprocessor.fit_scaler(sample_data)
                    logger.info("Fitted MinMaxScaler for backtesting")

                for i in range(lookback, len(prices) - 5):
                    try:
                        X = preprocessor.prepare_prediction_data(df.iloc[i - lookback:i])
                        logger.info(f"Predicting for {symbol} at index {i}, X shape: {X.shape}")
                        prediction = model.predict(X, verbose=0)[0][0]
                        current_price = prices[i]
                        indicators = {
                            'sma_fast': sma_fast[i] if len(df) >= SMA_FAST_PERIOD and not pd.isna(sma_fast[i]) and sma_fast[i] > 0 else 0.0001,
                            'sma_slow': sma_slow[i] if len(df) >= SMA_SLOW_PERIOD and not pd.isna(sma_slow[i]) and sma_slow[i] > 0 else 0.0001,
                            'rsi': rsi[i] if len(df) >= RSI_PERIOD and not pd.isna(rsi[i]) and rsi[i] > 0 else 50.0,
                            'atr': atr[i] if len(df) >= ATR_PERIOD and not pd.isna(atr[i]) and atr[i] > 0 else 0.0001,
                            'price': current_price if not pd.isna(current_price) and current_price > 0 else 0.0001,
                            'upper_band': upper_band[i] if len(df) >= BOLINGER_PERIOD and not pd.isna(upper_band[i]) and upper_band[i] > 0 else 0.0001,
                            'lower_band': lower_band[i] if len(df) >= BOLINGER_PERIOD and not pd.isna(lower_band[i]) and lower_band[i] > 0 else 0.0001
                        }
                        if any(pd.isna(v) or v <= 0 for v in [indicators['price'], indicators['sma_fast'], indicators['sma_slow'], indicators['rsi'], indicators['atr'], indicators['upper_band'], indicators['lower_band']]):
                            logger.warning(f"Invalid or zero data at index {i} for {symbol}")
                            continue  # Properly nested within the for i loop

                        # Momentum Strategy
                        momentum_indicators = indicators.copy()
                        momentum_threshold = PRED_THRESHOLD
                        if (prediction > momentum_threshold and 
                            any(momentum_indicators['sma_fast'] > momentum_indicators['sma_slow'] for tf in ['1m', '5m', '15m']) and 
                            any(rsi[i - j] < 50 for j in range(3)) and  # Short-term RSI check
                            indicators['price'] > indicators['upper_band'] and 
                            not position['momentum']):
                            quantity = self.trading_bot.adjust_quantity(symbol, indicators['price'])
                            step_size = self.trading_bot.lot_sizes.get(symbol, 0.0001)  # Use lot_sizes from trading_bot
                            if step_size and step_size > 0:
                                quantity = round(quantity / step_size) * step_size
                            if quantity > 0 and balance >= (indicators['price'] * quantity):
                                symbol_trades['momentum'] += 1
                                total_trades += 1
                                position['momentum'] = quantity
                                buy_price['momentum'] = indicators['price']
                                balance -= indicators['price'] * quantity
                                log_trade(symbol, 'BUY', indicators['price'], quantity, balance=balance)
                                trades_log.append({
                                    'symbol': symbol, 'strategy': 'momentum', 'action': 'BUY', 
                                    'price': indicators['price'], 'quantity': quantity, 'balance': balance
                                })
                                logger.info(f"Backtest trade for {symbol} #{symbol_trades['momentum']} (Momentum): Buy at {buy_price['momentum']:.6f}, Balance: {balance:.2f} USDT")

                        # Mean Reversion Strategy
                        mean_reversion_indicators = indicators.copy()
                        sma_slow_series = df['SMA_Slow'].dropna()
                        if len(sma_slow_series) > SMA_SLOW_PERIOD and np.std(sma_slow_series) > 0:
                            mean_reversion_indicators['z_score'] = (indicators['price'] - mean_reversion_indicators['sma_slow']) / np.std(sma_slow_series)
                        else:
                            mean_reversion_indicators['z_score'] = 0.0
                        if not position['mean_reversion'] and abs(mean_reversion_indicators['z_score']) > 0.6 and (mean_reversion_indicators['z_score'] < -0.6 or mean_reversion_indicators['z_score'] > 0.6):
                            mean_reversion_threshold = PRED_THRESHOLD * 0.7
                            if (prediction > mean_reversion_threshold and 
                                any(rsi[i - j] > 55 or rsi[i - j] < 45 for j in range(3)) and 
                                not position['mean_reversion']):
                                quantity = self.trading_bot.adjust_quantity(symbol, indicators['price'])
                                step_size = self.trading_bot.lot_sizes.get(symbol, 0.0001)
                                if step_size and step_size > 0:
                                    quantity = round(quantity / step_size) * step_size
                                if quantity > 0 and balance >= (indicators['price'] * quantity):
                                    symbol_trades['mean_reversion'] += 1
                                    total_trades += 1
                                    position['mean_reversion'] = quantity
                                    buy_price['mean_reversion'] = indicators['price']
                                    balance -= indicators['price'] * quantity
                                    log_trade(symbol, 'BUY', indicators['price'], quantity, balance=balance)
                                    trades_log.append({
                                        'symbol': symbol, 'strategy': 'mean_reversion', 'action': 'BUY', 
                                        'price': indicators['price'], 'quantity': quantity, 'balance': balance
                                    })
                                    logger.info(f"Backtest trade for {symbol} #{symbol_trades['mean_reversion']} (Mean Reversion): Buy at {buy_price['mean_reversion']:.6f}, Balance: {balance:.2f} USDT")

                        # Volatility Strategy
                        volatility_indicators = indicators.copy()
                        if not position['volatility'] and volatility_indicators['atr'] > 0.0001:
                            volatility_threshold = PRED_THRESHOLD * 0.6
                            if (prediction > volatility_threshold and 
                                volatility_indicators['price'] > volatility_indicators['upper_band'] and 
                                not position['volatility']):
                                quantity = self.trading_bot.adjust_quantity(symbol, indicators['price'])
                                step_size = self.trading_bot.lot_sizes.get(symbol, 0.0001)
                                if step_size and step_size > 0:
                                    quantity = round(quantity / step_size) * step_size
                                if quantity > 0 and balance >= (indicators['price'] * quantity):
                                    symbol_trades['volatility'] += 1
                                    total_trades += 1
                                    position['volatility'] = quantity
                                    buy_price['volatility'] = indicators['price']
                                    balance -= indicators['price'] * quantity
                                    log_trade(symbol, 'BUY', indicators['price'], quantity, balance=balance)
                                    trades_log.append({
                                        'symbol': symbol, 'strategy': 'volatility', 'action': 'BUY', 
                                        'price': indicators['price'], 'quantity': quantity, 'balance': balance
                                    })
                                    logger.info(f"Backtest trade for {symbol} #{symbol_trades['volatility']} (Volatility): Buy at {buy_price['volatility']:.6f}, Balance: {balance:.2f} USDT")

                        # Sell logic for all strategies, using TradingBot's method if available
                        for strategy in strategies:
                            if position[strategy] and buy_price[strategy] > 0 and indicators['price'] > 0:
                                try:
                                    dynamic_stop_loss, dynamic_take_profit = self.trading_bot.calculate_dynamic_limits(indicators['atr'], strategy)
                                except AttributeError:
                                    # Fallback if calculate_dynamic_limits is not implemented
                                    dynamic_stop_loss = STOP_LOSS_BASE
                                    dynamic_take_profit = TAKE_PROFIT_BASE
                                    logger.warning(f"calculate_dynamic_limits not found in trading_bot for {strategy}. Using default limits: Stop Loss {dynamic_stop_loss*100}%, Take Profit {dynamic_take_profit*100}%")

                                profit_pct = (indicators['price'] - buy_price[strategy]) / buy_price[strategy] * 100 if buy_price[strategy] > 0 else 0.0
                                if (profit_pct >= dynamic_take_profit * 100 or 
                                    profit_pct <= dynamic_stop_loss * 100 or 
                                    (indicators['price'] - buy_price[strategy]) / buy_price[strategy] <= -indicators['atr'] * 1.5):
                                    balance += position[strategy] * indicators['price']
                                    log_trade(symbol, 'SELL', indicators['price'], position[strategy], balance=balance, profit=profit_pct * balance / 100 if buy_price[strategy] > 0 else 0.0)
                                    trades_log.append({
                                        'symbol': symbol, 'strategy': strategy, 'action': 'SELL', 
                                        'price': indicators['price'], 'quantity': position[strategy], 'balance': balance, 'profit_pct': profit_pct
                                    })
                                    if profit_pct >= dynamic_take_profit * 100:
                                        profits.append(profit_pct)
                                    elif profit_pct <= dynamic_stop_loss * 100:
                                        profits.append(profit_pct)
                                    position[strategy] = 0
                                    buy_price[strategy] = 0.0

                    except Exception as e:
                        logger.error(f"Error in backtest for {symbol} at index {i}: {e}")
                        continue  # Properly nested within the except block

            for strategy in strategies:
                logger.info(f"Completed backtest for {symbol} with {symbol_trades[strategy]} {strategy.capitalize()} trades")

            # Log performance metrics for this symbol
            symbol_trades_log = [t for t in trades_log if t['symbol'] == symbol]
            win_rate = sum(1 for t in symbol_trades_log if t['profit_pct'] > 0) / len(symbol_trades_log) if symbol_trades_log else 0.0
            avg_profit = np.mean([t['profit_pct'] for t in symbol_trades_log]) if symbol_trades_log else 0.0
            logger.info(f"Performance for {symbol}: Win Rate {win_rate:.2%}, Avg Profit {avg_profit:.2f}%")

        avg_profit = sum(profits) / len(profits) if profits else 0.0
        sharpe = calculate_sharpe_ratio(profits, RISK_FREE_RATE) if profits else 0.0
        var = calculate_var(profits) if profits else 0.0
        logger.info(f"Backtest completed. Total trades: {total_trades}, Average Profit: {avg_profit:.2f}%, Sharpe Ratio: {sharpe:.2f}, VaR (95%): {var:.2f}% with {balance:.2f} USDT")
        return avg_profit