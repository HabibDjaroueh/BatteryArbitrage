"""
Forecast error analysis: perfect foresight vs forecast-driven DAM revenue.
Quantifies the dollar cost of forecast uncertainty. Zero Streamlit — pure Python/pandas/numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.soc_engine import BatterySpec, SoCEngine, DispatchAction
from src.forecaster import BATTERY_LZ_COL


# ── Perfect Foresight Benchmark ────────────────────────────────────────────

def run_perfect_foresight(
    actual_prices: pd.Series,
    spec:          BatterySpec,
) -> pd.DataFrame:
    """
    Computes the perfect foresight upper bound for a single day.

    The battery knows exact prices in advance and selects the single
    best charge window and discharge window of duration_hours each.
    Causal constraint: discharge window must start after charge window ends.
    """
    n_hours = int(spec.duration_hours)
    prices  = actual_prices.values
    index   = actual_prices.index

    best_revenue  = 0.0
    best_c_hours  = []
    best_d_hours  = []

    for c_start in range(24 - n_hours):
        c_end   = c_start + n_hours
        c_hours = list(range(c_start, c_end))
        avg_c   = float(np.mean(prices[c_hours]))

        for d_start in range(c_end, 24 - n_hours + 1):
            d_end   = d_start + n_hours
            d_hours = list(range(d_start, d_end))
            avg_d   = float(np.mean(prices[d_hours]))

            est = (
                avg_d * spec.power_mw * spec.discharge_eff * n_hours
                - avg_c * spec.power_mw * n_hours
            )
            if est > best_revenue:
                best_revenue = est
                best_c_hours = c_hours
                best_d_hours = d_hours

    engine = SoCEngine(spec)
    rows   = []

    for h, ts in enumerate(index):
        if h >= 24:
            break
        is_charge    = h in best_c_hours
        is_discharge = h in best_d_hours

        action = DispatchAction(
            charge_mw=    spec.power_mw if is_charge    else 0.0,
            discharge_mw= spec.power_mw if is_discharge else 0.0,
        )
        step = engine.step(action)

        price             = float(prices[h])
        charge_cost       = price * step["energy_charged_mwh"]
        discharge_revenue = price * step["energy_delivered_mwh"]

        rows.append({
            "timestamp":            ts,
            "hour":                 h,
            "charge_mw":            spec.power_mw if is_charge    else 0.0,
            "discharge_mw":         spec.power_mw if is_discharge else 0.0,
            "actual_price":         round(price, 2),
            "energy_charged_mwh":   step["energy_charged_mwh"],
            "energy_delivered_mwh": step["energy_delivered_mwh"],
            "charge_cost":          round(-charge_cost, 2),
            "discharge_revenue":    round(discharge_revenue, 2),
            "net_revenue":          round(discharge_revenue - charge_cost, 2),
            "soc_pct":              step["soc_pct_after"],
        })

    return pd.DataFrame(rows).set_index("timestamp")


def run_perfect_foresight_campaign(
    actual_df: pd.DataFrame,
    spec:      BatterySpec,
) -> pd.DataFrame:
    """
    Runs perfect foresight dispatch for every day in the dataset.
    Returns one row per day: date, pf_net_revenue, pf_charge_cost, pf_discharge_revenue.
    """
    if BATTERY_LZ_COL not in actual_df.columns:
        raise ValueError(f"{BATTERY_LZ_COL} not found in actual_df")

    daily_results   = []
    operating_days  = sorted(set(actual_df.index.date))

    for day in operating_days:
        day_data = actual_df[actual_df.index.date == day]
        if len(day_data) < 24:
            continue

        actual_prices = day_data[BATTERY_LZ_COL]

        try:
            pf_df = run_perfect_foresight(actual_prices, spec)
        except Exception as e:
            print(f"⚠️  Perfect foresight failed for {day}: {e}")
            continue

        daily_results.append({
            "date":                 day,
            "pf_net_revenue":       round(pf_df["net_revenue"].sum(), 2),
            "pf_charge_cost":       round(pf_df["charge_cost"].sum(), 2),
            "pf_discharge_revenue": round(pf_df["discharge_revenue"].sum(), 2),
        })

    return pd.DataFrame(daily_results)


# ── Forecast Error Analysis ────────────────────────────────────────────────

def compute_forecast_error_analysis(
    pf_campaign:  pd.DataFrame,
    dam_campaign: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges perfect foresight and DAM campaign results and computes
    forecast error cost and capture ratio for every day.
    """
    pf  = pf_campaign.copy()
    dam = dam_campaign.copy()

    pf["date"]  = pd.to_datetime(pf["date"]).dt.date
    dam["date"] = pd.to_datetime(dam["date"]).dt.date

    merged = pd.merge(pf, dam, on="date", how="inner")

    merged["forecast_error_cost"] = (
        merged["pf_net_revenue"] - merged["net_revenue"]
    ).round(2)

    merged["capture_ratio"] = merged.apply(
        lambda r: (
            round(r["net_revenue"] / r["pf_net_revenue"] * 100, 1)
            if r["pf_net_revenue"] != 0 else 0.0
        ),
        axis=1,
    )

    merged["idle_day"] = merged["net_revenue"] == 0.0

    return merged[[
        "date",
        "pf_net_revenue",
        "net_revenue",
        "forecast_error_cost",
        "capture_ratio",
        "idle_day",
    ]].rename(columns={"net_revenue": "dam_revenue"})


# ── Headline Summary ───────────────────────────────────────────────────────

def forecast_error_summary(
    analysis_df: pd.DataFrame,
    spec:        BatterySpec,
) -> dict:
    """
    Headline summary statistics from the forecast error analysis.
    """
    n_days = len(analysis_df)
    years  = n_days / 365.25

    total_pf    = analysis_df["pf_net_revenue"].sum()
    total_dam   = analysis_df["dam_revenue"].sum()
    total_error = analysis_df["forecast_error_cost"].sum()

    overall_capture = (
        total_dam / total_pf * 100
        if total_pf != 0 else 0.0
    )

    worst_idx  = analysis_df["forecast_error_cost"].idxmax()
    worst_day  = analysis_df.loc[worst_idx, "date"]
    worst_cost = analysis_df.loc[worst_idx, "forecast_error_cost"]

    below_50  = (analysis_df["capture_ratio"] < 50).sum()
    idle_days = analysis_df["idle_day"].sum()

    pf_per_mw_yr  = total_pf / spec.power_mw / years if years > 0 else 0
    dam_per_mw_yr = total_dam / spec.power_mw / years if years > 0 else 0
    err_per_mw_yr = total_error / spec.power_mw / years if years > 0 else 0

    return {
        "total_pf_revenue":          round(total_pf, 0),
        "total_dam_revenue":         round(total_dam, 0),
        "total_forecast_error_cost": round(total_error, 0),
        "overall_capture_ratio":     round(overall_capture, 1),

        "avg_daily_pf_revenue":      round(analysis_df["pf_net_revenue"].mean(), 2),
        "avg_daily_dam_revenue":     round(analysis_df["dam_revenue"].mean(), 2),
        "avg_daily_error_cost":      round(analysis_df["forecast_error_cost"].mean(), 2),
        "avg_capture_ratio":         round(analysis_df["capture_ratio"].mean(), 1),
        "p25_capture_ratio":         round(analysis_df["capture_ratio"].quantile(0.25), 1),
        "p75_capture_ratio":         round(analysis_df["capture_ratio"].quantile(0.75), 1),

        "worst_error_cost_day":      str(worst_day),
        "worst_error_cost":          round(worst_cost, 2),
        "days_capture_below_50pct":  int(below_50),
        "idle_days":                 int(idle_days),

        "pf_revenue_per_mw_year":    round(pf_per_mw_yr, 0),
        "dam_revenue_per_mw_year":   round(dam_per_mw_yr, 0),
        "error_cost_per_mw_year":    round(err_per_mw_yr, 0),
    }


# ── Monthly Breakdown ──────────────────────────────────────────────────────

def monthly_error_breakdown(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates forecast error analysis by calendar month.
    """
    df = analysis_df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

    monthly = df.groupby("month").agg(
        pf_revenue=          ("pf_net_revenue", "sum"),
        dam_revenue=         ("dam_revenue", "sum"),
        forecast_error_cost= ("forecast_error_cost", "sum"),
        avg_capture_ratio=   ("capture_ratio", "mean"),
        days=                ("date", "count"),
    ).reset_index()

    monthly["month"] = monthly["month"].astype(str)

    return monthly.rename(columns={
        "month":               "Month",
        "pf_revenue":          "PF Revenue ($)",
        "dam_revenue":         "DAM Revenue ($)",
        "forecast_error_cost": "Forecast Error Cost ($)",
        "avg_capture_ratio":   "Avg Capture Ratio (%)",
        "days":                "Days",
    }).round(2)


# ── Distribution Analysis ──────────────────────────────────────────────────

def capture_ratio_distribution(analysis_df: pd.DataFrame) -> dict:
    """
    Distribution of daily capture ratios and regime classification.
    """
    cr = analysis_df["capture_ratio"]

    return {
        "p10": round(float(cr.quantile(0.10)), 1),
        "p25": round(float(cr.quantile(0.25)), 1),
        "p50": round(float(cr.quantile(0.50)), 1),
        "p75": round(float(cr.quantile(0.75)), 1),
        "p90": round(float(cr.quantile(0.90)), 1),

        "pct_days_excellent": round((cr > 90).mean() * 100, 1),
        "pct_days_good":      round(((cr >= 70) & (cr <= 90)).mean() * 100, 1),
        "pct_days_moderate":  round(((cr >= 50) & (cr < 70)).mean() * 100, 1),
        "pct_days_poor":      round(((cr >= 0) & (cr < 50)).mean() * 100, 1),
        "pct_days_negative":  round((cr < 0).mean() * 100, 1),
    }
