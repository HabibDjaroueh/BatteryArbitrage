"""
DAM bidding and settlement at LZ_HOUSTON_DAM only. Spreads never in revenue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.soc_engine import BatterySpec, SoCEngine, DispatchAction
from src.forecaster import BATTERY_LZ_COL, DAM_CUTOFF_HOUR


SETTLEMENT_COL = BATTERY_LZ_COL


# ── DAM Bid ────────────────────────────────────────────────────────────────

class DAMBid:
    """
    One day's DAM self-schedule for the Houston battery.
    Bid based on FORECAST LZ_HOUSTON_DAM; settlement at ACTUAL LZ_HOUSTON_DAM.
    """

    def __init__(
        self,
        operating_date:   pd.Timestamp,
        submission_time:  pd.Timestamp,
        charge_hours:     list[int],
        discharge_hours:  list[int],
        battery_mw:       float,
        forecast_prices:  dict[int, float],
    ):
        assert submission_time.date() < operating_date.date(), (
            "Bid must be submitted the day before the operating day"
        )
        assert submission_time.hour < DAM_CUTOFF_HOUR, (
            f"Bid must be submitted before {DAM_CUTOFF_HOUR}:00 AM CPT"
        )
        assert len(set(charge_hours) & set(discharge_hours)) == 0, (
            "Charge and discharge windows cannot overlap"
        )
        assert all(isinstance(h, int) and 0 <= h <= 23 for h in charge_hours), (
            "All charge hours must be integers 0–23"
        )
        assert all(isinstance(h, int) and 0 <= h <= 23 for h in discharge_hours), (
            "All discharge hours must be integers 0–23"
        )
        if charge_hours and discharge_hours:
            assert min(discharge_hours) > max(charge_hours), (
                "Discharge window must start after charge window ends"
            )

        self.operating_date   = operating_date
        self.submission_time  = submission_time
        self.charge_hours     = sorted(charge_hours)
        self.discharge_hours  = sorted(discharge_hours)
        self.battery_mw       = battery_mw
        self.forecast_prices  = forecast_prices

    def to_schedule(self) -> pd.DataFrame:
        hours = pd.date_range(
            self.operating_date.normalize(), periods=24, freq="h"
        )
        rows = []
        for h, ts in enumerate(hours):
            rows.append({
                "timestamp":           ts,
                "hour":                h,
                "charge_mw":           self.battery_mw if h in self.charge_hours else 0.0,
                "discharge_mw":        self.battery_mw if h in self.discharge_hours else 0.0,
                "forecast_lz_houston": self.forecast_prices.get(h),
            })
        return pd.DataFrame(rows).set_index("timestamp")


# ── Bid Optimiser ──────────────────────────────────────────────────────────

def optimise_dam_bid(
    forecast_prices: pd.Series,
    spec:           BatterySpec,
    operating_date: pd.Timestamp,
) -> DAMBid:
    """
    Optimal charge/discharge windows using FORECASTED LZ_HOUSTON_DAM only.
    """
    n_hours = int(spec.duration_hours)
    submission = operating_date.normalize() - pd.Timedelta(days=1)
    submission = submission.replace(hour=9, minute=0)

    best_revenue   = 0.0
    best_charge    = []
    best_discharge = []

    # Ensure forecast_prices has integer index 0..23
    if not all(i in forecast_prices.index for i in range(24)):
        fp = pd.Series(forecast_prices.values[:24], index=range(min(24, len(forecast_prices))))
    else:
        fp = forecast_prices.reindex(range(24)).fillna(0)

    for c_start in range(24 - n_hours + 1):
        c_end   = c_start + n_hours
        c_hours = list(range(c_start, c_end))
        avg_c   = float(fp.iloc[c_hours].mean())

        for d_start in range(c_end, 24 - n_hours + 1):
            d_end   = d_start + n_hours
            d_hours = list(range(d_start, d_end))
            avg_d   = float(fp.iloc[d_hours].mean())

            est_revenue = (
                avg_d * spec.power_mw * spec.discharge_eff * n_hours
                - avg_c * spec.power_mw * n_hours
            )

            if est_revenue > best_revenue:
                best_revenue   = est_revenue
                best_charge    = c_hours
                best_discharge = d_hours

    forecast_dict = {h: round(float(fp.iloc[h]), 2) for h in range(24)}

    return DAMBid(
        operating_date=operating_date,
        submission_time=submission,
        charge_hours=best_charge,
        discharge_hours=best_discharge,
        battery_mw=spec.power_mw,
        forecast_prices=forecast_dict,
    )


# ── Settlement ─────────────────────────────────────────────────────────────

def settle_dam_bid(
    bid:           DAMBid,
    actual_prices: pd.Series,
    spec:          BatterySpec,
) -> dict:
    """
    Settles the DAM bid at actual LZ_HOUSTON_DAM prices.
    Revenue = actual_lz_houston × energy; spreads do not appear.
    """
    schedule_df = bid.to_schedule()
    engine      = SoCEngine(spec)
    hourly_results = []

    for ts, row in schedule_df.iterrows():
        action = DispatchAction(
            charge_mw=    max(0.0, float(row["charge_mw"])),
            discharge_mw= max(0.0, float(row["discharge_mw"])),
        )
        step = engine.step(action)

        ap = actual_prices.get(ts, np.nan)
        actual_price = 0.0 if pd.isna(ap) else float(ap)
        charge_cost       = actual_price * step["energy_charged_mwh"]
        discharge_revenue = actual_price * step["energy_delivered_mwh"]
        net_hour_revenue  = discharge_revenue - charge_cost

        hourly_results.append({
            "timestamp":           ts,
            "hour":                ts.hour,
            "charge_mw_bid":       row["charge_mw"],
            "discharge_mw_bid":    row["discharge_mw"],
            "charge_mw_actual":    step["charge_mw_actual"],
            "discharge_mw_actual": step["discharge_mw_actual"],
            "forecast_lz_houston": row["forecast_lz_houston"],
            "actual_lz_houston":   round(actual_price, 2),
            "energy_charged_mwh":  step["energy_charged_mwh"],
            "energy_delivered_mwh": step["energy_delivered_mwh"],
            "charge_cost":        round(-charge_cost, 2),
            "discharge_revenue":  round(discharge_revenue, 2),
            "net_revenue":        round(net_hour_revenue, 2),
            "soc_pct":            step["soc_pct_after"],
            "clipped":            step["charge_clipped"] or step["discharge_clipped"],
        })

    result_df = pd.DataFrame(hourly_results).set_index("timestamp")

    gross_charge_cost       = result_df["charge_cost"].sum()
    gross_discharge_revenue = result_df["discharge_revenue"].sum()
    net_revenue             = result_df["net_revenue"].sum()

    forecast_charge_cost = sum(
        bid.forecast_prices.get(h, 0) * spec.power_mw
        for h in bid.charge_hours
    )
    forecast_discharge_revenue = sum(
        bid.forecast_prices.get(h, 0) * spec.power_mw * spec.discharge_eff
        for h in bid.discharge_hours
    )
    forecast_revenue = forecast_discharge_revenue - forecast_charge_cost

    return {
        "operating_date":          bid.operating_date,
        "schedule_df":             result_df,
        "gross_charge_cost":       round(gross_charge_cost, 2),
        "gross_discharge_revenue": round(gross_discharge_revenue, 2),
        "net_revenue":             round(net_revenue, 2),
        "forecast_revenue":        round(forecast_revenue, 2),
        "realisation_gap":         round(net_revenue - forecast_revenue, 2),
        "charge_hours":            bid.charge_hours,
        "discharge_hours":         bid.discharge_hours,
        "was_profitable":          net_revenue > 0,
        "soc_violations":          int(result_df["clipped"].sum()),
    }


# ── Multi-Day DAM Campaign ─────────────────────────────────────────────────

def run_dam_campaign(
    forecast_df: pd.DataFrame,
    actual_df:   pd.DataFrame,
    spec:        BatterySpec,
) -> pd.DataFrame:
    """
    Historical DAM backtest: for each day, optimise bid on forecast
    LZ_HOUSTON_DAM, settle at actual LZ_HOUSTON_DAM.
    """
    if "forecast_houston_price" not in forecast_df.columns:
        raise ValueError(
            "forecast_df must contain 'forecast_houston_price'. "
            "Run generate_forecast_range() first."
        )
    if SETTLEMENT_COL not in actual_df.columns:
        raise ValueError(
            f"actual_df must contain '{SETTLEMENT_COL}' (LZ_HOUSTON_DAM)."
        )

    operating_days = sorted(set(forecast_df.index.date))
    daily_results = []

    for day in operating_days:
        op_date  = pd.Timestamp(day)
        day_mask = forecast_df.index.date == day
        day_fcst = forecast_df.loc[day_mask]

        if len(day_fcst) < 24:
            continue

        forecast_prices = pd.Series(
            day_fcst["forecast_houston_price"].values,
            index=range(len(day_fcst)),
        )

        bid = optimise_dam_bid(
            forecast_prices=forecast_prices,
            spec=spec,
            operating_date=op_date,
        )

        actual_day    = actual_df[actual_df.index.date == day]
        actual_prices = actual_day[SETTLEMENT_COL]

        if len(actual_prices) == 0:
            continue

        settlement = settle_dam_bid(bid, actual_prices, spec)

        daily_results.append({
            "date":                    op_date.date(),
            "net_revenue":             settlement["net_revenue"],
            "forecast_revenue":        settlement["forecast_revenue"],
            "realisation_gap":         settlement["realisation_gap"],
            "charge_hours":            str(settlement["charge_hours"]),
            "discharge_hours":         str(settlement["discharge_hours"]),
            "was_profitable":          settlement["was_profitable"],
            "soc_violations":          settlement["soc_violations"],
            "gross_charge_cost":       settlement["gross_charge_cost"],
            "gross_discharge_revenue": settlement["gross_discharge_revenue"],
        })

    return pd.DataFrame(daily_results)


# ── Campaign KPIs ──────────────────────────────────────────────────────────

def dam_campaign_kpis(
    campaign_df: pd.DataFrame,
    spec:        BatterySpec,
) -> dict:
    """Headline KPIs from a completed DAM campaign. All revenue at LZ_HOUSTON_DAM."""
    total_days = len(campaign_df)
    years      = total_days / 365.25

    total_revenue    = campaign_df["net_revenue"].sum()
    avg_daily        = campaign_df["net_revenue"].mean()
    best_day         = campaign_df["net_revenue"].max()
    worst_day        = campaign_df["net_revenue"].min()
    profitable_pct   = campaign_df["was_profitable"].mean() * 100
    total_violations = campaign_df["soc_violations"].sum()
    total_gap        = campaign_df["realisation_gap"].sum()
    rev_per_mw_year  = total_revenue / spec.power_mw / years if years > 0 else 0

    fcast_total   = campaign_df["forecast_revenue"].sum()
    capture_ratio = (
        total_revenue / fcast_total * 100
        if fcast_total != 0 else 0.0
    )

    return {
        "total_net_revenue":     round(total_revenue, 0),
        "avg_daily_revenue":     round(avg_daily, 2),
        "best_day":              round(best_day, 2),
        "worst_day":             round(worst_day, 2),
        "profitable_days_pct":   round(profitable_pct, 1),
        "total_soc_violations":  int(total_violations),
        "total_realisation_gap": round(total_gap, 0),
        "avg_capture_ratio_pct": round(capture_ratio, 1),
        "revenue_per_mw_year":   round(rev_per_mw_year, 0),
    }
