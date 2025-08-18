import os
import numpy as np
import pandas as pd
from joblib import dump, load
from datetime import timedelta
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(r"C:\Users\Babblu\OneDrive\Documents\GitHub\Stock_Analysis.AI\src\forex_predictor.py"))))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import TARGET_FX
from src.database import fetch_fx_news, fetch_fx_prices

# Map event to whether a higher actual is USD-positive (+1) or USD-negative (-1)
POS_IF_HIGH = {
    "non-farm employment change": +1,
    "average hourly earnings": +1,
    "cpi": +1, "core cpi": +1, "ppi": +1, "retail sales": +1, "gdp": +1,
    "ism manufacturing pmi": +1, "ism services pmi": +1,
}
POS_IF_LOW = {
    "unemployment rate": +1,            # lower unemployment is USD-positive
    "initial jobless claims": +1,
    "continuing jobless claims": +1,
}

IMPACT_W = {"Low":1.0, "Medium":2.0, "High":3.5}

def _usd_direction_for_pair(symbol: str):
    # If pair starts with USD (e.g., USDINR) -> USD strength = price UP
    # If pair ends with USD (e.g., EURUSD)  -> USD strength = price DOWN
    base = symbol.upper().split("=")[0]
    return +1 if base.startswith("USD") else -1

def _event_sign(event_name: str):
    e = event_name.lower()
    for k in POS_IF_HIGH:
        if k in e: return +1
    for k in POS_IF_LOW:
        if k in e: return -1
    return +1  # default: higher better for USD

def _to_num(x):
    if x is None: return np.nan
    try:
        s = str(x).replace("%","").replace(",","").strip()
        return float(s)
    except Exception:
        return np.nan

def _build_daily_news_features(df_news):
    if df_news.empty:
        return pd.DataFrame(columns=["date","usd_news_weighted","n_events","n_high","n_med","n_low"])

    df = df_news.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["date"] = df["event_time"].dt.floor("D")

    # numeric fields
    df["actual_n"]   = df["actual"].apply(_to_num)
    df["forecast_n"] = df["forecast"].apply(_to_num)

    # surprise (percentage if forecast exists)
    df["surprise"] = np.where(
        df["forecast_n"].notna() & (df["forecast_n"]!=0),
        (df["actual_n"] - df["forecast_n"]) / np.abs(df["forecast_n"]),
        df["actual_n"] - df["forecast_n"]
    )

    # direction sign per event
    df["dir_sign"] = df["event"].apply(_event_sign)

    # impact weight
    df["w"] = df["impact"].map(IMPACT_W).fillna(2.0)

    # weighted contribution to USD strength (positive -> USD bullish)
    df["usd_contrib"] = df["surprise"].fillna(0.0) * df["dir_sign"] * df["w"]

    daily = df.groupby("date").agg(
        usd_news_weighted=("usd_contrib","sum"),
        n_events=("event","count"),
        n_high=("impact", lambda s: (s=="High").sum()),
        n_med=("impact", lambda s: (s=="Medium").sum()),
        n_low=("impact", lambda s: (s=="Low").sum()),
    ).reset_index()

    return daily

def _build_price_daily(df_prices):
    df = df_prices.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    daily = pd.DataFrame()
    daily["open"] = df["open"].resample("1D").first()
    daily["high"] = df["high"].resample("1D").max()
    daily["low"]  = df["low"].resample("1D").min()
    daily["close"]= df["close"].resample("1D").last()
    daily["ret_1d"] = daily["close"].pct_change()
    daily["vol_5"] = daily["ret_1d"].rolling(5).std()
    daily["ma_5"] = daily["close"].rolling(5).mean()
    daily["ma_10"] = daily["close"].rolling(10).mean()
    daily.dropna(inplace=True)
    daily = daily.reset_index().rename(columns={"datetime":"date"})
    return daily

def _prepare_dataset(symbol=None):
    symbol = symbol or TARGET_FX
    price = fetch_fx_prices(symbol)
    news  = fetch_fx_news("USD")

    if price.empty:
        raise ValueError("No FX price data found. Run fx_scraper first.")
    price_d = _build_price_daily(price)
    news_d  = _build_daily_news_features(news)

    df = price_d.merge(news_d, on="date", how="left")
    df[["usd_news_weighted","n_events","n_high","n_med","n_low"]] = \
        df[["usd_news_weighted","n_events","n_high","n_med","n_low"]].fillna(0.0)

    # target: next-day USD strength direction mapped to price movement
    usd_dir = _usd_direction_for_pair(symbol)
    df = df.sort_values("date")
    df["close_next"] = df["close"].shift(-1)
    # price up means usd_dir * (close_next - close) > 0 -> USD stronger
    df["usd_up"] = (usd_dir * (df["close_next"] - df["close"]) > 0).astype(int)
    df = df.dropna(subset=["close_next"])

    features = [
        "open","high","low","close","ret_1d","vol_5","ma_5","ma_10",
        "usd_news_weighted","n_events","n_high","n_med","n_low"
    ]
    X = df[features].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    y = df["usd_up"].astype(int)
    dates = df["date"]
    return X, y, dates, features

def train_fx(symbol=None, n_splits=5, save=True):
    symbol = symbol or TARGET_FX
    X, y, dates, feats = _prepare_dataset(symbol)
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X)//30)))
    oof = np.zeros(len(y))
    models = []
    for i, (tr, va) in enumerate(tscv.split(X)):
        clf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=4, n_jobs=-1, random_state=100+i)
        clf.fit(X.iloc[tr], y.iloc[tr])
        proba = clf.predict_proba(X.iloc[va])[:,1]
        oof[va] = proba
        acc = accuracy_score(y.iloc[va], (proba>0.5).astype(int))
        try: auc = roc_auc_score(y.iloc[va], proba)
        except ValueError: auc = np.nan
        print(f"[FOLD {i+1}] ACC={acc:.3f} AUC={auc if not np.isnan(auc) else 'NA'}")
        models.append(clf)

    final = RandomForestClassifier(n_estimators=600, min_samples_split=3, n_jobs=-1, random_state=2025)
    final.fit(X, y)
    if save:
        os.makedirs("models", exist_ok=True)
        path = f"models/FX_{symbol.replace('=','')}.joblib"
        dump({"model": final, "features": feats, "symbol": symbol}, path)
        print(f"[INFO] Saved {path}")

    oof_acc = accuracy_score(y, (oof>0.5).astype(int))
    try: oof_auc = roc_auc_score(y, oof)
    except ValueError: oof_auc = np.nan
    print(f"[OOF] ACC={oof_acc:.3f} AUC={oof_auc if not np.isnan(oof_auc) else 'NA'}")
    return final, feats

def predict_fx(symbol=None):
    symbol = symbol or TARGET_FX
    path = f"models/FX_{symbol.replace('=','')}.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Train the FX model first.")

    bundle = load(path)
    model = bundle["model"]; feats = bundle["features"]
    X, y, dates, _ = _prepare_dataset(symbol)
    last_x = X.iloc[[-1]][feats]
    last_date = dates.iloc[-1]
    proba = float(model.predict_proba(last_x)[:,1][0])
    pred = int(proba>0.5)
    print(f"[FX PRED] {symbol} | last_date={last_date.date()} → USD stronger next day? {bool(pred)} (p={proba:.3f})")
    return {"date": str(last_date.date()), "usd_stronger_next_day": bool(pred), "prob": proba}

if __name__ == "__main__":
    train_fx()
    predict_fx()
