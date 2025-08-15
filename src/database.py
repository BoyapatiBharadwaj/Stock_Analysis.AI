# src/database.py
import mysql.connector
from src.config import MYSQL_CONFIG
import pandas as pd

def get_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

def insert_stock_data(symbol, datetime, open_p, high_p, low_p, close_p, volume):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO stocks (symbol, datetime, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (symbol, datetime, open_p, high_p, low_p, close_p, volume))
    conn.commit()
    cursor.close()
    conn.close()


def insert_news_data(symbol, headline, source, sentiment_score, sentiment_label, datetime):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO news (symbol, headline, source, sentiment_score, sentiment_label, datetime)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (symbol, headline, source, sentiment_score, sentiment_label, datetime))
    conn.commit()
    cursor.close()
    conn.close()


def fetch_stock_data(symbol, start=None, end=None):
    conn = get_connection()
    q = """
        SELECT symbol, datetime, open, high, low, close, volume
        FROM stocks
        WHERE symbol=%s
        {start}
        {end}
        ORDER BY datetime ASC
    """.format(
        start="AND datetime >= %s" if start else "",
        end="AND datetime <= %s" if end else ""
    )
    params = [symbol]
    if start: params.append(start)
    if end: params.append(end)
    df = pd.read_sql(q, conn, params=params)
    conn.close()
    return df

def fetch_news_data(symbol, start=None, end=None):
    conn = get_connection()
    q = """
        SELECT symbol, headline, source, sentiment_score, sentiment_label, datetime
        FROM news
        WHERE symbol=%s
        {start}
        {end}
        ORDER BY datetime ASC
    """.format(
        start="AND datetime >= %s" if start else "",
        end="AND datetime <= %s" if end else ""
    )
    params = [symbol]
    if start: params.append(start)
    if end: params.append(end)
    df = pd.read_sql(q, conn, params=params)
    conn.close()
    return df