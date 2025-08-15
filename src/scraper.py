# src/scraper.py
import yfinance as yf
from datetime import datetime
from src.database import insert_stock_data

def fetch_and_store_stock(symbol="AAPL"):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")  # 1 day, 1-min interval

    for index, row in data.iterrows():
        insert_stock_data(
            symbol,
            index.to_pydatetime(),
            row["Open"],
            row["High"],
            row["Low"],
            row["Close"],
            int(row["Volume"])
        )
    print(f"[INFO] Stored {len(data)} records for {symbol}.")

if __name__ == "__main__":
    fetch_and_store_stock("AAPL")
