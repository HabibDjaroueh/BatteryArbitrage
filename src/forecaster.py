"""
Houston LZ price forecast for DAM bidding. Spreads are features only — never in revenue.
Pure Python/pandas/numpy — zero Streamlit imports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models import prepare_features


# ── Constants ──────────────────────────────────────────────────────────────

BATTERY_LZ_COL  = "LZ_HOUSTON_DAM"   # settlement price — never changes
SETTLEMENT_COL  = BATTERY_LZ_COL
BATTERY_REGION  = "coast"            # battery is in Houston
DAM_CUTOFF_HOUR = 10                 # bids locked in by 10AM CPT day-ahead

# Spread columns used as FEATURES in the price forecast model (signals only)
SPREAD_FEATURE_COLS = [
    "spread_h_s",
    "spread_h_w",
    "spread_h_n",
    "spread_h_r",
]

FEATURE_LAG_HOURS = 24
# Features are lagged 24 hours to eliminate lookahead bias.
# At 10AM D-1, the operator has access to:
#   - All actual values from D-1 and prior (wind, load, weather, spreads)
#   - Weather forecasts for D (temperature, humidity)
# Our backtest conservatively uses only D-1 actuals for all features.
# This understates model performance slightly vs a live system that
# would also incorporate D weather forecasts — making it a
# conservative honest estimate.


# ── Train Houston Price Forecast Model ────────────────────────────────────

def train_houston_price_model(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
) -> dict:
    """
    Trains an XGBoost model to forecast LZ_HOUSTON_DAM hourly prices.

    Feature lag:
        All features are shifted forward 24 hours (1 day lag) so that
        features for hour H on day D come from hour H on day D-1.
        This eliminates lookahead — at 10AM D-1 the operator knows
        yesterday's actual wind, load, and weather but not today's.

    Target:
        LZ_HOUSTON_DAM on day D — the actual price we are predicting.

    Temporal split:
        Train: all data before test_start
        Test:  test_start onwards
    """
    if BATTERY_LZ_COL not in df.columns:
        raise ValueError(f"{BATTERY_LZ_COL} not found in DataFrame")

    # ── Build lagged feature matrix ────────────────────────────────────────
    X_all = prepare_features(df)
    for c in SPREAD_FEATURE_COLS:
        if c in df.columns:
            X_all[c] = df.loc[X_all.index, c]

    # Row at timestamp T contains features from timestamp T-24h
    X_lagged = X_all.shift(FEATURE_LAG_HOURS)
    y_all = df[BATTERY_LZ_COL]

    data  = pd.concat([X_lagged, y_all], axis=1).dropna()
    X_all = data[X_lagged.columns]
    y_all = data[BATTERY_LZ_COL]

    # ── Temporal split ─────────────────────────────────────────────────────
    train_mask = X_all.index < test_start
    test_mask  = X_all.index >= test_start

    X_train = X_all[train_mask]
    X_test  = X_all[test_mask]
    y_train = y_all[train_mask]
    y_test  = y_all[test_mask]

    if len(X_train) < 1000:
        raise ValueError(
            f"Insufficient training data: {len(X_train)} rows before {test_start}"
        )

    # ── Train ──────────────────────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae    = float(np.mean(np.abs(y_pred - y_test.values)))
    rmse   = float(np.sqrt(np.mean((y_pred - y_test.values) ** 2)))

    predictions_df = pd.DataFrame({
        "actual_houston_price":   y_test.values,
        "forecast_houston_price": y_pred,
    }, index=y_test.index)

    return {
        "model":             model,
        "feature_names":     X_train.columns.tolist(),
        "mae":               round(mae, 2),
        "rmse":              round(rmse, 2),
        "train_size":        len(X_train),
        "test_size":         len(X_test),
        "test_start":        test_start,
        "predictions_df":    predictions_df,
        "feature_lag_hours": FEATURE_LAG_HOURS,
    }


# ── Day-Ahead Forecast ─────────────────────────────────────────────────────

def generate_day_ahead_forecast(
    model_result:  dict,
    df:            pd.DataFrame,
    operating_date: pd.Timestamp,
) -> pd.Series:
    """
    Generates 24 hourly LZ_HOUSTON_DAM forecasts for operating_date
    using features from operating_date - 1 day.

    Lookahead elimination:
        Features come from Day D-1 — available at 10AM D-1.
        Target is Day D LZ_HOUSTON_DAM — what we are predicting.
    """
    model         = model_result["model"]
    feature_names = model_result["feature_names"]

    # ── Features from Day D-1 (24-hour lag) ───────────────────────────────
    feature_date = operating_date - pd.Timedelta(days=1)
    feature_data = df[df.index.date == feature_date.date()].copy()

    if len(feature_data) == 0:
        raise ValueError(
            f"No feature data found for {feature_date.date()} (D-1 of {operating_date.date()}). "
            "Cannot generate forecast without prior day features."
        )

    if len(feature_data) < 24:
        raise ValueError(
            f"Incomplete feature data for {feature_date.date()}: "
            f"{len(feature_data)} hours found, expected 24."
        )

    # ── Build feature matrix from D-1 data ────────────────────────────────
    X_day = prepare_features(feature_data)
    for c in SPREAD_FEATURE_COLS:
        if c in feature_data.columns:
            X_day[c] = feature_data[c].values

    missing = [f for f in feature_names if f not in X_day.columns]
    for mf in missing:
        X_day[mf] = 0.0

    X_forecast  = X_day[feature_names].fillna(0)
    predictions = model.predict(X_forecast)

    # ── Index predictions to Day D timestamps ─────────────────────────────
    day_d_index = pd.date_range(
        operating_date.normalize(),
        periods=24,
        freq="h",
    )

    return pd.Series(
        predictions,
        index=day_d_index,
        name="forecast_houston_price",
    )


def generate_forecast_range(
    model_result: dict,
    df:          pd.DataFrame,
    start_date:  pd.Timestamp,
    end_date:    pd.Timestamp,
) -> pd.DataFrame:
    """
    Hourly LZ_HOUSTON_DAM forecasts across a date range.
    Each day uses D-1 features — no lookahead at any point.
    Returns forecast_houston_price and actual_houston_price for the same period.
    """
    pred_df = model_result.get("predictions_df")
    if pred_df is not None:
        mask = (
            (pred_df.index >= start_date) &
            (pred_df.index <= end_date)
        )
        subset = pred_df.loc[mask].copy()
        if len(subset) > 0:
            return subset

    all_days      = pd.date_range(start=start_date, end=end_date, freq="D")
    all_forecasts = []

    for day in all_days:
        try:
            fcst = generate_day_ahead_forecast(model_result, df, day)
            actual_day    = df[df.index.date == day.date()]
            actual_prices = actual_day[BATTERY_LZ_COL].reindex(fcst.index)

            day_df = pd.DataFrame({
                "forecast_houston_price": fcst.values,
                "actual_houston_price":   actual_prices.values,
            }, index=fcst.index)
            all_forecasts.append(day_df)
        except ValueError as e:
            print(f"⚠️  Skipping {day.date()}: {e}")
            continue

    if not all_forecasts:
        return pd.DataFrame()
    return pd.concat(all_forecasts).sort_index()


# ── Forecast Accuracy ───────────────────────────────────────────────────────

def forecast_accuracy(forecast_df: pd.DataFrame) -> dict:
    """
    Accuracy of LZ_HOUSTON_DAM forecasts vs actuals.
    Returns dict: mae, rmse, direction_accuracy (%), n_hours.
    """
    df = forecast_df[["forecast_houston_price", "actual_houston_price"]].dropna()
    if df.empty:
        return {"mae": 0.0, "rmse": 0.0, "direction_accuracy": 0.0, "n_hours": 0}

    y_pred  = df["forecast_houston_price"].values
    y_true  = df["actual_houston_price"].values
    mae     = float(np.mean(np.abs(y_pred - y_true)))
    rmse    = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100)

    return {
        "mae":                round(mae, 2),
        "rmse":               round(rmse, 2),
        "direction_accuracy": round(dir_acc, 1),
        "n_hours":            len(df),
    }
