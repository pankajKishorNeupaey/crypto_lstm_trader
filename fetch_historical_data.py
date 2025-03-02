# fetch_historical_data.py
from fetch_data import DataFetcher
from config import SYMBOLS

def fetch_and_save_historical_data():
    fetcher = DataFetcher()
    for symbol in SYMBOLS:
        df = fetcher.get_historical_data(symbol, limit=2000, use_cache=False)
        if df is not None and not df.empty:
            print(f"Fetched historical data for {symbol}")
        else:
            print(f"Failed to fetch historical data for {symbol} - check logs/data_fetch.log for details")

if __name__ == "__main__":
    fetch_and_save_historical_data()