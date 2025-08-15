# streamlit run src/dashboard.py

import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(r"C:\Users\Babblu\OneDrive\Documents\GitHub\Stock_Analysis.AI\src\dashboard.py"))))


# Local imports
# The app is inside src/, so relative imports work when running: `streamlit run src/dashboard.py`
from src.database import fetch_stock_data, fetch_news_data
from src.predictor import predict_next_day, train_model

st.set_page_config(
    page_title="AI Stock Sentiment & Prediction Dashboard",
    layout="wide",
)

# -------------------------
# Sidebar Controls
# -------------------------
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA",
    "TCS.NS", "RELIANCE.NS", "INFY.NS"
]

st.sidebar.title("Controls")
symbol = st.sidebar.selectbox("Stock Symbol", DEFAULT_SYMBOLS, index=0)

# Date range for charts (uses dates present in DB)
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=60)
start = st.sidebar.date_input("Start Date", start_date)
end = st.sidebar.date_input("End Date", end_date)
if start > end:
    st.sidebar.error("Start date must be before end date")

st.sidebar.markdown("---")
retrain = st.sidebar.button("🔁 Retrain Model")
predict = st.sidebar.button("📈 Predict Next Day")

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh data (every 5 min)", value=False)

# -------------------------
# Cached Data Loaders
# -------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(sym: str, start: datetime | str | None, end: datetime | str | None) -> pd.DataFrame:
    df = fetch_stock_data(sym, start=str(start) if start else None, end=str(end) if end else None)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_news_data(sym: str, start: datetime | str | None, end: datetime | str | None) -> pd.DataFrame:
    df = fetch_news_data(sym, start=str(start) if start else None, end=str(end) if end else None)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    return df

# -------------------------
# Helper: Aggregate sentiment daily
# -------------------------
def aggregate_daily_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame(columns=["date","sent_mean","sent_count","pos_ratio","neg_ratio","neu_ratio"])    
    tmp = df_news.copy()
    tmp["date"] = tmp["datetime"].dt.floor("D")
    # counts
    counts = (
        tmp.pivot_table(index="date", columns="sentiment_label", values="headline", aggfunc="count")
           .fillna(0)
           .rename(columns={"positive":"cnt_positive","neutral":"cnt_neutral","negative":"cnt_negative"})
    )
    agg = tmp.groupby("date")["sentiment_score"].agg(["mean","count"]).rename(columns={"mean":"sent_mean","count":"sent_count"})
    out = agg.join(counts, how="left").fillna(0)
    total = out[["cnt_positive","cnt_neutral","cnt_negative"]].sum(axis=1).replace(0, np.nan)
    out["pos_ratio"] = out["cnt_positive"]/total
    out["neg_ratio"] = out["cnt_negative"]/total
    out["neu_ratio"] = out["cnt_neutral"]/total
    out = out.drop(columns=["cnt_positive","cnt_neutral","cnt_negative"]).reset_index()
    return out

# -------------------------
# Header
# -------------------------
st.title("📊 AI Stock Sentiment & Prediction Dashboard")
st.caption("Live prices from your DB + news sentiment + next‑day direction prediction")

# Optional auto-refresh
if auto_refresh:
    st.experimental_singleton.clear()  # noop for Streamlit >=1.29 (kept for compatibility)
    st.experimental_rerun

# -------------------------
# Data Loading
# -------------------------
prices = load_price_data(symbol, start, end + timedelta(days=1))
news = load_news_data(symbol, start, end + timedelta(days=1))
sent_daily = aggregate_daily_sentiment(news)

col_price, col_sent = st.columns([2, 1])

with col_price:
    st.subheader(f"Price – {symbol}")
    if prices.empty:
        st.warning("No price data in DB for the selected period. Run src/scraper.py first.")
    else:
        fig_price = px.line(prices, x="datetime", y="close", title=f"{symbol} Close Price")
        fig_price.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
        st.plotly_chart(fig_price, use_container_width=True)

with col_sent:
    st.subheader("Daily Sentiment")
    if sent_daily.empty:
        st.info("No news/sentiment found in this range. Run src/news_scraper.py.")
    else:
        fig_sent = px.bar(sent_daily, x="date", y="sent_mean", title="Avg Sentiment (compound)")
        fig_sent.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
        st.plotly_chart(fig_sent, use_container_width=True)

# -------------------------
# Prediction + Actions
# -------------------------
st.markdown("---")
left, mid, right = st.columns([1.2, 1, 1])

with left:
    st.subheader("Next‑Day Prediction")
    if predict:
        try:
            res = predict_next_day(symbol)
            proba_up = res["p_up"]
            pred_label = "UP" if res["pred_up"] == 1 else "DOWN"
            badge = "🟢" if pred_label == "UP" else "🔴"
            st.success(f"{badge} {symbol} next‑day: **{pred_label}** (p_up={proba_up:.2f})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.caption("Click ‘Predict Next Day’ in the sidebar to generate a fresh prediction using the saved model.")

with mid:
    st.subheader("Retrain Model")
    if retrain:
        with st.spinner("Training model… this may take a minute"):
            try:
                train_model(symbol)
                st.success("Model trained & saved to /models")
            except Exception as e:
                st.error(f"Training failed: {e}")
    else:
        st.caption("Use after new data is ingested. Models saved under /models/<SYMBOL>_clf.joblib")

with right:
    st.subheader("Data Coverage")
    cov_msg = []
    if prices.empty:
        cov_msg.append("❌ Prices")
    else:
        cov_msg.append(f"✅ Prices ({prices['datetime'].min().date()} ➜ {prices['datetime'].max().date()})")
    if news.empty:
        cov_msg.append("❌ News/Sentiment")
    else:
        cov_msg.append(f"✅ News ({news['datetime'].min().date()} ➜ {news['datetime'].max().date()})")
    st.write("\n".join(cov_msg))

# -------------------------
# Recent Headlines
# -------------------------
st.markdown("---")
st.subheader("📰 Recent Headlines & Sentiment")

if news.empty:
    st.info("No recent headlines for this symbol in the selected range.")
else:
    # Show the 20 most recent
    show = news.sort_values("datetime", ascending=False).head(20)
    def emoji(label: str) -> str:
        return {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(label, "⚪")

    for _, r in show.iterrows():
        st.markdown(
            f"{emoji(str(r['sentiment_label']))} **{r['headline']}**  "
            f"<br><span style='font-size:0.9em;color:gray'>{r['datetime']} — {r['source']}</span>",
            unsafe_allow_html=True,
        )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built with Streamlit · Data from your MySQL DB (stocks, news) · ML: RandomForest + sentiment features")
