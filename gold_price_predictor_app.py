"""
Gold Price Predictor App (Machine Learning)

Commodity: Gold Futures (GC=F) from Yahoo Finance via yfinance.
Goal: Predict NEXT trading day's Close price using supervised learning.

Model: RandomForestRegressor (decision-tree ensemble)
Why: Works well on tabular engineered features, handles non-linear patterns, and is a decision-tree-based method.

Outputs:
- Prints dataset record count and feature count
- Prints evaluation metrics (MAE, RMSE, R^2)
- Prints next-day prediction
- Saves plot: gold_prediction_plot.png
"""

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


@dataclass
class Config:
    ticker: str = "GC=F"
    start: str = "2015-01-01"
    end: str = None              # None = up to most recent available
    test_size: float = 0.20      # last 20% for test (time-based split)
    random_state: int = 42
    n_estimators: int = 400
    max_depth: int = 10
    min_samples_leaf: int = 2


def download_data(cfg: Config) -> pd.DataFrame:
    """Download daily historical data using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("Missing dependency: yfinance")
        print("Install it with: pip install yfinance")
        sys.exit(1)

    df = yf.download(cfg.ticker, start=cfg.start, end=cfg.end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned. Check ticker or internet connection.")

    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)

    # Keep common OHLCV fields
    df = df[["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for time-series regression.
    Target is next day's close (shift -1).
    """
    data = df.copy()

    # Lag features
    data["close_lag1"] = data["Close"].shift(1)
    data["close_lag2"] = data["Close"].shift(2)

    # Return features
    data["ret_1d"] = data["Close"].pct_change()

    # Rolling stats
    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_10"] = data["Close"].rolling(10).mean()
    data["ma_20"] = data["Close"].rolling(20).mean()

    data["vol_10"] = data["ret_1d"].rolling(10).std()
    data["vol_20"] = data["ret_1d"].rolling(20).std()

    # Price range / intraday change
    data["hl_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["oc_change"] = (data["Close"] - data["Open"]) / data["Open"]

    # Target: next day's close
    data["target_next_close"] = data["Close"].shift(-1)

    # Drop NaNs created by rolling/shift
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def train_test_split_time(data: pd.DataFrame, test_size: float):
    """Time-aware split: keep chronological order (no shuffling)."""
    n = len(data)
    split_idx = int(n * (1 - test_size))
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    return train, test


def train_model(cfg: Config, X_train, y_train):
    """Train RandomForestRegressor."""
    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return preds, mae, rmse, r2



def plot_results(dates, y_true, y_pred, out_path="gold_prediction_plot.png"):
    plt.figure()
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label="Predicted")
    plt.title("Gold Next-Day Close Prediction (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


def main():
    cfg = Config()

    print("=== Gold Price Predictor App ===")
    print(f"Ticker: {cfg.ticker} | Start: {cfg.start}")

    raw = download_data(cfg)
    data = make_features(raw)

    feature_cols = [
        "close_lag1", "close_lag2",
        "ret_1d", "ma_5", "ma_10", "ma_20",
        "vol_10", "vol_20",
        "hl_range", "oc_change",
        "Volume"
    ]

    # Print critical dataset details for your report
    print("\n--- Dataset Summary ---")
    print(f"Number of records (rows): {len(data)}")
    print(f"Number of features used : {len(feature_cols)}")
    print(f"Feature columns         : {feature_cols}")

    train, test = train_test_split_time(data, cfg.test_size)

    X_train = train[feature_cols]
    y_train = train["target_next_close"]

    X_test = test[feature_cols]
    y_test = test["target_next_close"]

    model = train_model(cfg, X_train, y_train)
    preds, mae, rmse, r2 = evaluate(model, X_test, y_test)

    print("\n--- Evaluation (Test Set) ---")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R^2 : {r2:,.4f}")

    # Next-day prediction from latest feature row
    latest_features = data.iloc[-1][feature_cols].to_frame().T
    next_close_pred = float(model.predict(latest_features)[0])

    last_date = data.iloc[-1]["date"]
    last_close = float(data.iloc[-1]["Close"])

    print("\n--- Next-Day Prediction ---")
    print(f"Last available date  : {last_date}")
    print(f"Last close           : {last_close:,.2f}")
    print(f"Predicted next close : {next_close_pred:,.2f}")

    plot_results(test["date"], y_test.values, preds, out_path="gold_prediction_plot.png")


if __name__ == "__main__":
    main()
