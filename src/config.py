# src/config.py

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",          # your MySQL username
    "password": "2535",  # your MySQL password
    "database": "stock_dashboard"
}


TARGET_FX = "USDINR=X"   # or "USDINR=X" if you prefer INR focus
FX_PRICE_INTERVAL = "1h" # "1d" | "1h" | "15m" (availability varies by ticker)
FF_TIMEZONE = "UTC"      # ForexFactory shows localizable times; we’ll treat as UTC

# polite scraping
SCRAPE_DELAY_SEC = 2
FF_BASE = "https://www.forexfactory.com"
