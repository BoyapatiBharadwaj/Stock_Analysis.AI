# src/news_scraper.py
import feedparser
from datetime import datetime
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(r"C:\Users\Babblu\OneDrive\Documents\GitHub\Stock_Analysis.AI\src\news_scraper.py"))))

from src.sentiment import analyze_sentiment
from src.database import insert_news_data

def fetch_and_store_news(symbol="AAPL"):
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        headline = entry.title
        source = entry.link
        published_time = datetime(*entry.published_parsed[:6])  # convert to datetime

        score, label = analyze_sentiment(headline)
        
        insert_news_data(symbol, headline, source, score, label, published_time)
    print(f"[INFO] Stored {len(feed.entries)} news articles for {symbol}.")

if __name__ == "__main__":
    fetch_and_store_news("AAPL")
