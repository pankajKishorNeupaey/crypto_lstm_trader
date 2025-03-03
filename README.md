# crypto_lstm_trader

This documentation, as of March 2, 2025, provides a complete guide to setup, usage, and maintenance, ensuring compatibility with Python 3.9, Binance Testnet, and the specified strategies.


Project Structure

The project is organized in the following directory structure:

text

WrapCopy

D:\PYPROJECTs\Trading BOts\crypto_lstm_trader\
│

├── data\

│   └── historical_data.csv  (Generated historical price data)

├── logs\

│   ├── data_fetch.log      (Logs for data fetching)

│   ├── trading.log         (Logs for trading activities)

│   ├── trading_debug.log   (Detailed debugging logs for trades)

│   └── training.log        (Logs for model training)

├── models\

│   ├── lstm_model.keras    (Trained LSTM model file)

│   └── scaler.pkl          (MinMaxScaler for data normalization)

│

├── config.py               (Configuration settings)

├── fetch_data.py           (Data fetching and indicator calculation)

├── fetch_historical_data.py (Script to fetch and save historical data)

├── preprocess.py           (Data preprocessing for LSTM)

├── trade.py                (Trading logic and strategies)

├── train.py                (Model training script)

├── utils.py                (Utility functions for logging and backtesting)

└── README.md               (This documentation)


Prerequisites

Before setting up the project, ensure you have:

Python 3.9 or higher installed ("C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\python.exe")
 Binance Testnet API key and secret (replace placeholders in config.py)
 Optional: NewsAPI key for sentiment analysis (replace placeholder in config.py)
