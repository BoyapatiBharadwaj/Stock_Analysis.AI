# src/scraper.py
import yfinance as yf
from datetime import datetime
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(r"C:\Users\Babblu\OneDrive\Documents\GitHub\Stock_Analysis.AI\src\scraper.py"))))

from src.database import insert_stock_data

def fetch_and_store_stock(symbol="AAPL"):
    stock = yf.Ticker(symbol)
    # Pull 3 years of daily historical data
    data = stock.history(period="3y", interval="1d")

    for index, row in data.iterrows():
        insert_stock_data(
            symbol,
            index.to_pydatetime(),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"])
        )
    print(f"[INFO] Stored {len(data)} daily records for {symbol}.")

if __name__ == "__main__":
    fetch_and_store_stock("AAPL")
