import time
import logging
import mysql.connector
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ---------------- MySQL Connection ----------------
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",         # change to your MySQL username
        password="password", # change to your MySQL password
        database="forex_db"  # change to your DB name
    )

# ---------------- Scraper ----------------
def scrape_forex_events():
    url = "https://www.forexfactory.com/calendar"

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service("C:/chromedriver/chromedriver.exe")  # change path if needed
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)

    time.sleep(5)  # wait for JS to load

    events = []
    rows = driver.find_elements(By.CSS_SELECTOR, "tr.calendar__row")

    logging.info(f"Found {len(rows)} rows in calendar")

    for row in rows:
        try:
            date = row.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__date").text.strip()
        except:
            date = ""

        try:
            time_val = row.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__time").text.strip()
        except:
            time_val = ""

        try:
            currency = row.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__currency").text.strip()
        except:
            currency = ""

        try:
            impact = row.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__impact").text.strip()
        except:
            impact = ""

        try:
            event = row.find_element(By.CSS_SELECTOR, ".calendar__event-title").text.strip()
        except:
            event = ""

        try:
            actual = row.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__actual").text.strip()
        except:
            actual = ""

        try:
            forecast = row.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__forecast").text.strip()
        except:
            forecast = ""

        try:
            previous = row.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__previous").text.strip()
        except:
            previous = ""

        if event:  # only save rows with valid events
            events.append((date, time_val, currency, impact, event, actual, forecast, previous))

    driver.quit()
    return events

# ---------------- Save to MySQL ----------------
def save_to_mysql(events):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forex_events (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date VARCHAR(50),
            time VARCHAR(50),
            currency VARCHAR(10),
            impact VARCHAR(50),
            event VARCHAR(255),
            actual VARCHAR(50),
            forecast VARCHAR(50),
            previous VARCHAR(50)
        )
    """)

    sql = """
        INSERT INTO forex_events 
        (date, time, currency, impact, event, actual, forecast, previous) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    cursor.executemany(sql, events)
    conn.commit()
    logging.info(f"Inserted {cursor.rowcount} forex events into DB.")

    cursor.close()
    conn.close()

# ---------------- Main ----------------
if __name__ == "__main__":
    events = scrape_forex_events()
    logging.info(f"Scraped {len(events)} events")

    if events:
        save_to_mysql(events)
