"""DEPRECATED — not used by any Streamlit page.

As of the forecast-driven DAM upgrade, the Battery Simulation page
(pages/4_Battery_Simulation.py) no longer calls run_arbitrage(),
run_rule_based(), or run_sensitivity(). No other page in pages/ imports
this module. These functions are retained for reference and for any
existing tests or scripts that still depend on them.

Current simulation flow (all in the UI):
  Step 1 — train_houston_price_model()     in src/forecaster.py
  Step 2 — generate_forecast_range()        in src/forecaster.py
  Step 3 — run_dam_campaign()               in src/dam_bidder.py
  Step 4 — run_perfect_foresight_campaign() in src/forecast_error.py
           compute_forecast_error_analysis() in src/forecast_error.py

Do not add new features here; extend forecaster, dam_bidder, or
forecast_error instead.
"""
import pandas as pd


# ── constants ──────────────────────────────────────────────────────────────

LZ_PRICE_COLS = {
    "coast": "LZ_HOUSTON_DAM",
    "south": "LZ_SOUTH_DAM",
    "west":  "LZ_WEST_DAM",
    "north": "LZ_NORTH_DAM",
    "east":  "LZ_RAYBN_DAM",
}

DEFAULT_CHARGE_PERCENTILE    = 25   # charge in bottom 25% of daily prices
DEFAULT_DISCHARGE_PERCENTILE = 75   # discharge in top 25% of daily prices


# ── core simulation ────────────────────────────────────────────────────────

def run_arbitrage(
    df: pd.DataFrame,
    lz_col: str,
    battery_mw: float,
    duration_hours: float,
    efficiency: float,
) -> pd.DataFrame:
    """
    Simulates a perfect-foresight daily arbitrage strategy.

    Logic per day:
        - Charge at the single cheapest hour (min price)
        - Discharge at the single most expensive hour (max price)
        - Revenue = discharge_price × energy_out - charge_price × energy_in
        - energy_in  = battery_mw × 1 hour
        - energy_out = energy_in × efficiency (round-trip)
        - Only dispatch if discharge_price > charge_price (no loss days)
        - Charge hour must come before discharge hour (causal constraint)

    Parameters
    ----------
    df             : DataFrame with datetime index and lz_col price column
    lz_col         : LZ price column name e.g. "LZ_HOUSTON_DAM"
    battery_mw     : Battery power rating in MW
    duration_hours : Battery energy capacity = battery_mw × duration_hours MWh
    efficiency     : Round-trip efficiency as decimal e.g. 0.85

    Returns
    -------
    pd.DataFrame with one row per day:
        date, charge_hour, discharge_hour, charge_price, discharge_price,
        energy_charged_mwh, energy_discharged_mwh, gross_revenue,
        charge_cost, net_revenue, dispatched
    """

    prices = df[[lz_col]].copy()
    prices.index = pd.to_datetime(prices.index)
    prices["date"] = prices.index.date
    prices["hour"] = prices.index.hour

    energy_capacity = battery_mw * duration_hours  # MWh

    results = []

    for date, day in prices.groupby("date"):
        if len(day) < 2:
            continue

        # Find min and max price hours
        min_idx = day[lz_col].idxmin()
        max_idx = day[lz_col].idxmax()

        charge_hour    = day.loc[min_idx, "hour"]
        discharge_hour = day.loc[max_idx, "hour"]
        charge_price   = day.loc[min_idx, lz_col]
        discharge_price = day.loc[max_idx, lz_col]

        # Causal constraint — charge must come before discharge
        # If max price is before min price, find next best discharge hour
        if discharge_hour <= charge_hour:
            after_charge = day[day["hour"] > charge_hour]
            if after_charge.empty:
                results.append(_no_dispatch_row(date))
                continue
            max_idx         = after_charge[lz_col].idxmax()
            discharge_hour  = after_charge.loc[max_idx, "hour"]
            discharge_price = after_charge.loc[max_idx, lz_col]

        # Only dispatch if profitable
        if discharge_price <= charge_price:
            results.append(_no_dispatch_row(date))
            continue

        # Energy flows
        energy_charged    = min(battery_mw * 1.0, energy_capacity)   # MWh in
        energy_discharged = energy_charged * efficiency                # MWh out

        # Revenue
        charge_cost   = charge_price   * energy_charged      # cost to charge
        gross_revenue = discharge_price * energy_discharged   # revenue from discharge
        net_revenue   = gross_revenue - charge_cost

        results.append({
            "date":                  date,
            "charge_hour":           int(charge_hour),
            "discharge_hour":        int(discharge_hour),
            "charge_price":          round(float(charge_price),    2),
            "discharge_price":       round(float(discharge_price), 2),
            "price_spread":          round(float(discharge_price - charge_price), 2),
            "energy_charged_mwh":    round(float(energy_charged),    3),
            "energy_discharged_mwh": round(float(energy_discharged), 3),
            "gross_revenue":         round(float(gross_revenue), 2),
            "charge_cost":           round(float(charge_cost),   2),
            "net_revenue":           round(float(net_revenue),   2),
            "dispatched":            True,
        })

    return pd.DataFrame(results)


def run_rule_based(
    df: pd.DataFrame,
    lz_col: str,
    battery_mw: float,
    duration_hours: float,
    efficiency: float,
    charge_pct: int = 25,
    discharge_pct: int = 75,
) -> pd.DataFrame:
    """
    Rule-based battery dispatch strategy.

    Logic per day:
        1. Compute the charge threshold    = charge_pct    percentile of that day's prices
        2. Compute the discharge threshold = discharge_pct percentile of that day's prices
        3. Eligible charge hours    = hours where price <= charge threshold
        4. Eligible discharge hours = hours where price >= discharge threshold
        5. Charge in the cheapest eligible hour
        6. Discharge in the most expensive eligible hour AFTER the charge hour
        7. Apply efficiency and compute net revenue
        8. No dispatch if no valid charge/discharge pair found

    This is more realistic than perfect foresight — thresholds are derived
    from the day's own price distribution, simulating a operator who has
    a price forecast and sets dispatch rules accordingly.

    Parameters
    ----------
    df             : DataFrame with datetime index and lz_col
    lz_col         : LZ price column name
    battery_mw     : Battery power rating in MW
    duration_hours : Storage duration in hours
    efficiency     : Round-trip efficiency as decimal
    charge_pct     : Percentile threshold for charging (default 25)
    discharge_pct  : Percentile threshold for discharging (default 75)

    Returns
    -------
    Same schema as run_arbitrage() — one row per day with the same columns
    plus two additional columns:
        charge_threshold    : price threshold used for charging
        discharge_threshold : price threshold used for discharging
    """
    prices = df[[lz_col]].copy()
    prices.index = pd.to_datetime(prices.index)
    prices["date"] = prices.index.date
    prices["hour"] = prices.index.hour

    energy_capacity = battery_mw * duration_hours

    results = []

    for date, day in prices.groupby("date"):
        if len(day) < 2:
            continue

        charge_threshold    = day[lz_col].quantile(charge_pct    / 100)
        discharge_threshold = day[lz_col].quantile(discharge_pct / 100)

        # Eligible hours
        charge_hours    = day[day[lz_col] <= charge_threshold]
        discharge_hours = day[day[lz_col] >= discharge_threshold]

        if charge_hours.empty or discharge_hours.empty:
            results.append({
                **_no_dispatch_row(date),
                "charge_threshold":    round(float(charge_threshold),    2),
                "discharge_threshold": round(float(discharge_threshold), 2),
            })
            continue

        # Best charge hour — cheapest eligible
        charge_idx   = charge_hours[lz_col].idxmin()
        charge_hour  = charge_hours.loc[charge_idx, "hour"]
        charge_price = charge_hours.loc[charge_idx, lz_col]

        # Best discharge hour — most expensive eligible AFTER charge hour
        after_charge    = discharge_hours[discharge_hours["hour"] > charge_hour]
        if after_charge.empty:
            results.append({
                **_no_dispatch_row(date),
                "charge_threshold":    round(float(charge_threshold),    2),
                "discharge_threshold": round(float(discharge_threshold), 2),
            })
            continue

        discharge_idx   = after_charge[lz_col].idxmax()
        discharge_hour  = after_charge.loc[discharge_idx, "hour"]
        discharge_price = after_charge.loc[discharge_idx, lz_col]

        # Only dispatch if profitable
        if discharge_price <= charge_price:
            results.append({
                **_no_dispatch_row(date),
                "charge_threshold":    round(float(charge_threshold),    2),
                "discharge_threshold": round(float(discharge_threshold), 2),
            })
            continue

        # Energy flows
        energy_charged    = min(battery_mw * 1.0, energy_capacity)
        energy_discharged = energy_charged * efficiency

        charge_cost   = charge_price   * energy_charged
        gross_revenue = discharge_price * energy_discharged
        net_revenue   = gross_revenue - charge_cost

        results.append({
            "date":                  date,
            "charge_hour":           int(charge_hour),
            "discharge_hour":        int(discharge_hour),
            "charge_price":          round(float(charge_price),    2),
            "discharge_price":       round(float(discharge_price), 2),
            "price_spread":          round(float(discharge_price - charge_price), 2),
            "energy_charged_mwh":    round(float(energy_charged),    3),
            "energy_discharged_mwh": round(float(energy_discharged), 3),
            "gross_revenue":         round(float(gross_revenue), 2),
            "charge_cost":           round(float(charge_cost),   2),
            "net_revenue":           round(float(net_revenue),   2),
            "dispatched":            True,
            "charge_threshold":      round(float(charge_threshold),    2),
            "discharge_threshold":   round(float(discharge_threshold), 2),
        })

    return pd.DataFrame(results)


def _no_dispatch_row(date) -> dict:
    """Returns a zero-revenue row for days where dispatch is not profitable."""
    return {
        "date":                  date,
        "charge_hour":           None,
        "discharge_hour":        None,
        "charge_price":          None,
        "discharge_price":       None,
        "price_spread":          0.0,
        "energy_charged_mwh":    0.0,
        "energy_discharged_mwh": 0.0,
        "gross_revenue":         0.0,
        "charge_cost":           0.0,
        "net_revenue":           0.0,
        "dispatched":            False,
        "charge_threshold":      None,
        "discharge_threshold":   None,
    }


# ── aggregations ───────────────────────────────────────────────────────────

def monthly_revenue(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily simulation results to monthly totals.

    Returns
    -------
    pd.DataFrame with columns:
        Month, Dispatched Days, Total Net Revenue ($),
        Avg Daily Revenue ($), Avg Price Spread ($/MWh)
    """
    sim_df = sim_df.copy()
    sim_df["month"] = pd.to_datetime(sim_df["date"]).dt.to_period("M")

    monthly = sim_df.groupby("month").agg(
        dispatched_days=("dispatched",    "sum"),
        total_revenue=  ("net_revenue",   "sum"),
        avg_daily_rev=  ("net_revenue",   "mean"),
        avg_spread=     ("price_spread",  "mean"),
    ).reset_index()

    monthly["month"]         = monthly["month"].astype(str)
    monthly["total_revenue"] = monthly["total_revenue"].round(0)
    monthly["avg_daily_rev"] = monthly["avg_daily_rev"].round(0)
    monthly["avg_spread"]    = monthly["avg_spread"].round(2)

    monthly.columns = [
        "Month", "Dispatched Days",
        "Total Net Revenue ($)", "Avg Daily Revenue ($)",
        "Avg Price Spread ($/MWh)"
    ]

    return monthly


def simulation_kpis(
    sim_df: pd.DataFrame,
    battery_mw: float,
    duration_hours: float,
) -> dict:
    """
    Computes summary KPIs from the simulation results.

    Returns
    -------
    dict with:
        total_revenue, avg_daily_revenue, best_day_revenue,
        worst_day_revenue, dispatch_rate_pct, total_energy_mwh,
        revenue_per_mw_year, p90_daily_revenue
    """
    dispatched = sim_df[sim_df["dispatched"]]
    total_days = len(sim_df)

    total_revenue     = sim_df["net_revenue"].sum()
    avg_daily_revenue = sim_df["net_revenue"].mean()
    best_day_revenue  = sim_df["net_revenue"].max()
    worst_day_revenue = dispatched["net_revenue"].min() if len(dispatched) > 0 else 0
    dispatch_rate     = len(dispatched) / total_days * 100 if total_days > 0 else 0
    total_energy      = sim_df["energy_discharged_mwh"].sum()
    years             = total_days / 365.25
    rev_per_mw_year   = (total_revenue / battery_mw / years) if years > 0 else 0
    p90_daily         = sim_df["net_revenue"].quantile(0.90)

    return {
        "total_revenue":      round(total_revenue,     0),
        "avg_daily_revenue":  round(avg_daily_revenue, 2),
        "best_day_revenue":   round(best_day_revenue,  2),
        "worst_day_revenue":  round(worst_day_revenue, 2),
        "dispatch_rate_pct":  round(dispatch_rate,     1),
        "total_energy_mwh":   round(total_energy,      1),
        "revenue_per_mw_year":round(rev_per_mw_year,   0),
        "p90_daily_revenue":  round(p90_daily,         2),
    }


def run_sensitivity(
    df: pd.DataFrame,
    lz_col: str,
    battery_mw: float,
    efficiency_values: list[float],
    duration_values: list[int],
    strategy: str = "rule_based",
    charge_pct: int = 25,
    discharge_pct: int = 75,
) -> pd.DataFrame:
    """
    Runs a grid of simulations across efficiency and duration combinations.
    Returns a results matrix suitable for display as a heatmap or table.

    Parameters
    ----------
    df                : filtered regional DataFrame
    lz_col            : LZ price column
    battery_mw        : fixed power rating in MW
    efficiency_values : list of efficiency decimals e.g. [0.80, 0.85, 0.90]
    duration_values   : list of durations in hours e.g. [2, 4, 6]
    strategy          : "perfect_foresight" or "rule_based"
    charge_pct        : only used for rule_based strategy
    discharge_pct     : only used for rule_based strategy

    Returns
    -------
    pd.DataFrame — one row per (efficiency, duration) combination with columns:
        Efficiency (%), Duration (h), Total Revenue ($),
        Avg Daily Revenue ($), Revenue / MW / Year ($), Dispatch Rate (%)
    """
    rows = []

    for eff in efficiency_values:
        for dur in duration_values:
            try:
                if strategy == "perfect_foresight":
                    sim = run_arbitrage(df, lz_col, battery_mw, dur, eff)
                else:
                    sim = run_rule_based(
                        df, lz_col, battery_mw, dur, eff,
                        charge_pct, discharge_pct
                    )

                kpis = simulation_kpis(sim, battery_mw, dur)

                rows.append({
                    "Efficiency (%)":       int(eff * 100),
                    "Duration (h)":         int(dur),
                    "Total Revenue ($)":    kpis["total_revenue"],
                    "Avg Daily Rev ($)":    kpis["avg_daily_revenue"],
                    "Rev / MW / Year ($)":  kpis["revenue_per_mw_year"],
                    "Dispatch Rate (%)":    kpis["dispatch_rate_pct"],
                })
            except Exception:
                continue

    return pd.DataFrame(rows)
