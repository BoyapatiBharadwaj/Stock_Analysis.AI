# src/predictor.py
import os
from datetime import timedelta
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(r"C:\Users\Babblu\OneDrive\Documents\GitHub\Stock_Analysis.AI\src\predictor.py"))))


from src.database import fetch_stock_data, fetch_news_data

MODELS_DIR = "models"

def _rsi(close, window=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _build_daily_price_features(df_min):
    # Resample minute data to DAILY features
    df = df_min.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    daily = pd.DataFrame()
    daily["open"] = df["open"].resample("1D").first()
    daily["high"] = df["high"].resample("1D").max()
    daily["low"] = df["low"].resample("1D").min()
    daily["close"] = df["close"].resample("1D").last()
    daily["volume"] = df["volume"].resample("1D").sum()
    daily.dropna(inplace=True)

    # Technicals
    daily["ret_1d"] = daily["close"].pct_change()
    daily["ma_5"] = daily["close"].rolling(5).mean()
    daily["ma_10"] = daily["close"].rolling(10).mean()
    daily["ma_20"] = daily["close"].rolling(20).mean()
    daily["ma_ratio_5_20"] = daily["ma_5"] / daily["ma_20"]
    daily["vol_5"] = daily["ret_1d"].rolling(5).std()
    daily["vol_10"] = daily["ret_1d"].rolling(10).std()
    daily["rsi_14"] = _rsi(daily["close"], 14)
    daily["prev_close"] = daily["close"].shift(1)
    daily["gap_open"] = (daily["open"] - daily["prev_close"]) / daily["prev_close"]

    return daily

def _aggregate_daily_sentiment(df_news):
    if df_news.empty:
        # If no news yet, return empty to merge later
        return pd.DataFrame(columns=["date","sent_mean","sent_median","sent_count","pos_ratio","neg_ratio","neu_ratio"])
    df = df_news.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.floor("D")
    # counts by label
    counts = df.pivot_table(index="date", columns="sentiment_label", values="headline", aggfunc="count").fillna(0)
    counts.columns = [f"cnt_{c}" for c in counts.columns]
    # numeric aggregates
    agg = df.groupby("date")["sentiment_score"].agg(["mean", "median", "count"]).rename(
        columns={"mean":"sent_mean","median":"sent_median","count":"sent_count"}
    )
    out = agg.join(counts, how="left").fillna(0)
    total = out[["cnt_negative","cnt_neutral","cnt_positive"]].sum(axis=1).replace(0, np.nan)
    out["pos_ratio"] = out["cnt_positive"] / total
    out["neg_ratio"] = out["cnt_negative"] / total
    out["neu_ratio"] = out["cnt_neutral"] / total
    out = out.drop(columns=["cnt_negative","cnt_neutral","cnt_positive"])
    out = out.reset_index()
    return out

def _prepare_dataset(symbol):
    # Load
    price_min = fetch_stock_data(symbol)
    news = fetch_news_data(symbol)

    if price_min.empty:
        raise ValueError(f"No price data found for {symbol}. Run the scraper first.")

    # Features
    price_daily = _build_daily_price_features(price_min)
    sent_daily = _aggregate_daily_sentiment(news)

    price_daily = price_daily.reset_index().rename(columns={"datetime":"date"})
    df = price_daily.merge(sent_daily, on="date", how="left")

    # Fill sentiment gaps with 0 (no news that day)
    sent_cols = ["sent_mean","sent_median","sent_count","pos_ratio","neg_ratio","neu_ratio"]
    for c in sent_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = 0.0

    # Target: next-day direction (close_{t+1} > close_{t})
    df = df.sort_values("date")
    df["close_next"] = df["close"].shift(-1)
    df["target_up"] = (df["close_next"] > df["close"]).astype(int)

    # Drop last row without target
    df = df.dropna(subset=["close_next"])

    # Feature set
    feature_cols = [
        "open","high","low","close","volume","ret_1d","ma_5","ma_10","ma_20",
        "ma_ratio_5_20","vol_5","vol_10","rsi_14","gap_open",
        "sent_mean","sent_median","sent_count","pos_ratio","neg_ratio","neu_ratio"
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["target_up"].astype(int)
    dates = df["date"]
    return X, y, dates, feature_cols

def train_model(symbol="AAPL", save=True, n_splits=5, random_state=42):
    X, y, dates, feature_cols = _prepare_dataset(symbol)

    if len(X) < n_splits + 1:
        raise ValueError(
            f"Not enough samples to train {n_splits}-fold model. "
            f"Got only {len(X)} samples. "
            f"Try scraping more historical data for {symbol}."
        )

    # Time-series cross validation (keeps order)
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X)//30)))
    oof_preds = np.zeros(len(y))
    models = []

    for fold, (tr, va) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state + fold,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        models.append(clf)

        proba = clf.predict_proba(X_va)[:,1]
        oof_preds[va] = proba

        acc = accuracy_score(y_va, (proba > 0.5).astype(int))
        try:
            auc = roc_auc_score(y_va, proba)
        except ValueError:
            auc = np.nan
        print(f"[FOLD {fold+1}] ACC={acc:.3f} | AUC={auc if not np.isnan(auc) else 'NA'}")

    # Fit final model on all data
    final_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )
    final_clf.fit(X, y)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"{symbol}_clf.joblib")
        dump({"model": final_clf, "features": feature_cols}, path)
        print(f"[INFO] Saved model → {path}")

    # Report overall OOF metrics
    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    try:
        oof_auc = roc_auc_score(y, oof_preds)
    except ValueError:
        oof_auc = np.nan
    print(f"[OOF] ACC={oof_acc:.3f} | AUC={oof_auc if not np.isnan(oof_auc) else 'NA'}")
    return final_clf, feature_cols

def predict_next_day(symbol="AAPL"):
    # Load last available day’s features and predict t+1 direction
    model_path = os.path.join(MODELS_DIR, f"{symbol}_clf.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

    bundle = load(model_path)
    model = bundle["model"]
    feat_cols = bundle["features"]

    X, y, dates, _ = _prepare_dataset(symbol)
    last_x = X.iloc[[-1]][feat_cols]
    last_date = dates.iloc[-1]

    proba_up = float(model.predict_proba(last_x)[:,1][0])
    pred_up = int(proba_up > 0.5)
    print(f"[PRED] {symbol} | last_date={last_date.date()} → next_day_up={pred_up} (p_up={proba_up:.3f})")
    return {"date": str(last_date.date()), "pred_up": pred_up, "p_up": proba_up}

if __name__ == "__main__":
    # Quick CLI usage:
    # python src/predictor.py  (trains & predicts AAPL)
    sym = os.environ.get("SYMBOL", "AAPL")
    train_model(sym)
    predict_next_day(sym)
