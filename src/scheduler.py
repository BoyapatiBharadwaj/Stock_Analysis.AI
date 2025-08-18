# src/scheduler.py
import os
import sys
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(r"C:\Users\Babblu\OneDrive\Documents\GitHub\Stock_Analysis.AI\src\scheduler.py"))))

from src.scraper import fetch_and_store_stock_data
from src.news_scraper import fetch_and_store_news

from src.fx_scraper import fetch_and_store_fx_prices
from src.forex_factory_scraper import fetch_and_store_fx_news_history
from datetime import datetime, timedelta



scheduler = BlockingScheduler()

# FX prices every hour
@scheduler.scheduled_job('interval', hours=1)
def job_fx_prices():
    print("[SCHEDULER] Fetching FX prices...")
    fetch_and_store_fx_prices()

# USD news (previous day) once per day
@scheduler.scheduled_job('cron', hour=3)  # 03:00 UTC daily
def job_fx_news():
    print("[SCHEDULER] Scraping ForexFactory USD news (yesterday)...")
    y = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
    fetch_and_store_fx_news_history(y, y, currency="USD")



# Fetch stock price every 5 minutes
@scheduler.scheduled_job('interval', minutes=5)
def job_stock():
    print("[SCHEDULER] Fetching stock prices...")
    fetch_and_store_stock_data("AAPL")

# Fetch news every 30 minutes
@scheduler.scheduled_job('interval', minutes=30)
def job_news():
    print("[SCHEDULER] Fetching news headlines...")
    fetch_and_store_news("AAPL")

if __name__ == "__main__":
    print("[SCHEDULER] Starting...")
    scheduler.start()
