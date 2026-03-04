"""Risk metrics for battery arbitrage simulation results."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_risk_metrics(analysis_df: pd.DataFrame) -> dict:
    """
    Compute risk metrics from the forecast error analysis DataFrame.

    Parameters
    ----------
    analysis_df : DataFrame with columns: date, dam_revenue, pf_net_revenue,
                  forecast_error_cost, capture_ratio, idle_day

    Returns
    -------
    dict with risk metric values:
        - daily_var_5pct: 5th percentile daily revenue (VaR at 95% confidence)
        - cvar_5pct: Average revenue on worst 5% of days (Expected Shortfall)
        - max_drawdown: Largest peak-to-trough decline in cumulative revenue
        - max_drawdown_days: Duration of max drawdown in days
        - sharpe_ratio: Annualized risk-adjusted return
        - longest_losing_streak: Maximum consecutive days with zero or negative revenue
        - pct_profitable_days: Percentage of days with positive revenue
        - avg_daily_revenue: Mean daily revenue
        - std_daily_revenue: Standard deviation of daily revenue
        - skewness: Revenue distribution skewness
        - kurtosis: Revenue distribution kurtosis
    """
    rev = analysis_df["dam_revenue"].dropna()

    if rev.empty:
        return {
            k: 0.0
            for k in (
                "daily_var_5pct",
                "cvar_5pct",
                "max_drawdown",
                "max_drawdown_days",
                "sharpe_ratio",
                "longest_losing_streak",
                "pct_profitable_days",
                "avg_daily_revenue",
                "std_daily_revenue",
                "skewness",
                "kurtosis",
            )
        }

    # ── VaR (5th percentile) ────────────────────────────────────────────────
    var_5 = float(np.percentile(rev, 5))

    # ── CVaR / Expected Shortfall ────────────────────────────────────────────
    below_var = rev[rev <= var_5]
    cvar_5 = float(below_var.mean()) if len(below_var) > 0 else var_5

    # ── Maximum Drawdown ─────────────────────────────────────────────────────
    cumulative = rev.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())

    if max_dd < 0:
        dd_end_idx = drawdown.idxmin()
        peak_before = running_max.loc[:dd_end_idx].idxmax()
        dd_end_pos = analysis_df.index.get_loc(dd_end_idx)
        peak_pos = analysis_df.index.get_loc(peak_before)
        max_dd_days = int(dd_end_pos - peak_pos)
    else:
        max_dd_days = 0

    # ── Sharpe Ratio ─────────────────────────────────────────────────────────
    mean_rev = float(rev.mean())
    std_rev = float(rev.std())
    sharpe = (mean_rev / std_rev * np.sqrt(252)) if std_rev > 0 else 0.0

    # ── Longest Consecutive Losing Streak ────────────────────────────────────
    is_loss = (rev <= 0).values
    longest_streak = 0
    current_streak = 0
    for loss in is_loss:
        if loss:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

    pct_profitable = float((rev > 0).mean() * 100)

    return {
        "daily_var_5pct": var_5,
        "cvar_5pct": cvar_5,
        "max_drawdown": max_dd,
        "max_drawdown_days": max_dd_days,
        "sharpe_ratio": sharpe,
        "longest_losing_streak": longest_streak,
        "pct_profitable_days": pct_profitable,
        "avg_daily_revenue": mean_rev,
        "std_daily_revenue": std_rev,
        "skewness": float(rev.skew()),
        "kurtosis": float(rev.kurtosis()),
    }


def compute_seasonal_reliability(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly revenue statistics for seasonal reliability assessment.

    Parameters
    ----------
    analysis_df : DataFrame with columns: date, dam_revenue
    """
    df = analysis_df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    grouped = df.groupby("month")["dam_revenue"]
    stats = pd.DataFrame(
        {
            "Month": grouped.first().index,
            "Mean ($)": grouped.mean().values,
            "Std ($)": grouped.std().values,
            "Min ($)": grouped.min().values,
            "Max ($)": grouped.max().values,
            "Days": grouped.count().values,
        }
    )

    stats["Std ($)"] = stats["Std ($)"].fillna(0)
    stats["Reliable"] = (stats["Mean ($)"] > 0) & (
        stats["Std ($)"] < stats["Mean ($)"]
    )

    return stats


def compute_drawdown_series(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative revenue and drawdown series for charting.

    Parameters
    ----------
    analysis_df : DataFrame with columns: date, dam_revenue
    """
    df = pd.DataFrame(
        {
            "date": analysis_df["date"],
            "daily_revenue": analysis_df["dam_revenue"],
        }
    )
    df["cumulative_revenue"] = df["daily_revenue"].cumsum()
    df["running_max"] = df["cumulative_revenue"].cummax()
    df["drawdown"] = df["cumulative_revenue"] - df["running_max"]
    return df

