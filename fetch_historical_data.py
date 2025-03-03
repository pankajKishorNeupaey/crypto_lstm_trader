import pandas as pd
from fetch_data import DataFetcher
from config import SYMBOLS, HISTORICAL_DATA_PATH

def fetch_and_save_historical_data():
    """Fetch and save historical data for all symbols with enhanced features."""
    fetcher = DataFetcher()
    all_data = pd.DataFrame()
    successes = []
    failures = []

    for symbol in SYMBOLS:
        print(f"Fetching historical data for {symbol}...")
        df = fetcher.get_historical_data(symbol, limit=2000, use_cache=False)
        if df is not None and not df.empty:
            all_data = pd.concat([all_data, df], ignore_index=True)
            successes.append(symbol)
        else:
            failures.append(symbol)

    if not all_data.empty:
        all_data.to_csv(HISTORICAL_DATA_PATH, index=False)
        print(f"Saved historical data to {HISTORICAL_DATA_PATH}")
        print(f"Success: {successes}")
    else:
        print("No historical data fetched, check logs/data_fetch.log")
    if failures:
        print(f"Failed: {failures}")

if __name__ == "__main__":
    fetch_and_save_historical_data()