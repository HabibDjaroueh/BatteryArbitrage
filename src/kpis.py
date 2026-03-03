from __future__ import annotations

from typing import Dict

import pandas as pd

# Maps spread letter to zone name
SPREAD_LETTER_TO_ZONE = {
    "h": "coast",
    "s": "south",
    "w": "west",
    "n": "north",
    "r": "east",
}

# Maps zone to its net load column
ZONE_NET_LOAD_MAP = {
    "coast": "COAST_Net_Load",
    "south": "SOUTH_Net_Load",
    "west":  "WEST_Net_Load",
    "north": "NORTH_Net_Load",
    "east":  "EAST_Net_Load",
}

# Maps zone to its load column (for RE penetration fallback)
ZONE_LOAD_MAP = {
    "coast": "COAST_Load",
    "south": "SOUTH_Load",
    "west":  "WEST_Load",
    "north": "NORTH_Load",
    "east":  "EAST_Load",
}

ZONE_WIND_COL = {
    "coast": "wind_gen_coast_mw",
    "south": "wind_gen_south_mw",
    "west":  "wind_gen_west_mw",
    "north": "wind_gen_north_mw",
    "east":  "wind_gen_east_mw",
}

ZONE_SOLAR_COL = {
    "coast": "solar_gen_coast_mw",
    "south": "solar_gen_south_mw",
    "west":  "solar_gen_west_mw",
    "north": "solar_gen_north_mw",
    "east":  "solar_gen_east_mw",
}

ZONE_LABEL = {
    "coast": "Houston",
    "south": "South",
    "west":  "West",
    "north": "North",
    "east":  "East (Rayburn)",
}


def premium_and_discount_zones(spread_col: str) -> tuple[str | None, str | None]:
    """
    Derives the premium (left) and discount (right) zone from a spread
    column name.

    e.g. spread_h_w → premium = "coast" (Houston), discount = "west"
         spread_r_n → premium = "east"  (Rayburn), discount = "north"

    Returns
    -------
    (premium_zone, discount_zone) as region name strings
    """
    parts         = spread_col.replace("spread_", "").split("_")
    premium_zone  = SPREAD_LETTER_TO_ZONE.get(parts[0]) if len(parts) >= 1 else None
    discount_zone = SPREAD_LETTER_TO_ZONE.get(parts[1]) if len(parts) >= 2 else None
    return premium_zone, discount_zone


def zones_from_spreads(spread_cols: list[str]) -> list[str]:
    """
    Extracts the ordered unique list of zones involved in the
    selected spread columns.

    e.g. ["spread_h_w", "spread_h_n"] → ["coast", "west", "north"]
         ["spread_h_s"]               → ["coast", "south"]

    Parameters
    ----------
    spread_cols : list of spread column names e.g. ["spread_h_w"]

    Returns
    -------
    list of zone names in order of first appearance, deduplicated
    """
    zones = []
    for spread_col in spread_cols:
        parts = spread_col.replace("spread_", "").split("_")
        for letter in parts:
            zone = SPREAD_LETTER_TO_ZONE.get(letter)
            if zone and zone not in zones:
                zones.append(zone)
    return zones


def compute_kpis(df: pd.DataFrame, spread_col: str) -> dict:
    """
    Computes 5 trader-relevant KPIs for a given spread column.

    KPIs:
        avg_spread      : Mean spread value in filtered window
        volatility      : 30D rolling std of daily mean spread
        capture_rate    : % of hours where spread > 0 (positive direction held)
        re_curtailment  : Zone-specific (wind + solar) / gross load × 100 on discount zone
        net_load_stress : Net load as % of gross load on premium zone

    Parameters
    ----------
    df         : filtered regional DataFrame — must contain all zone columns
    spread_col : spread column name e.g. "spread_h_s"

    Returns
    -------
    dict of formatted strings
    """
    if spread_col not in df.columns:
        return {
            "avg_spread":      "N/A",
            "volatility":      "N/A",
            "capture_rate":    "N/A",
            "re_curtailment":  "N/A",
            "net_load_stress": "N/A",
        }

    spread  = df[spread_col].dropna()
    premium_zone, discount_zone = premium_and_discount_zones(spread_col)

    # ── Avg Spread ─────────────────────────────────────────────────────────
    avg_spread = spread.mean()

    # ── 30D Volatility ─────────────────────────────────────────────────────
    volatility = spread.resample("D").mean().rolling(30).std().mean()

    # ── Spread Capture Rate ────────────────────────────────────────────────
    capture_rate = (spread > 0).mean() * 100

    # ── RE Curtailment Pressure — zone-specific ───────────────────────────
    # (wind + solar) / gross load × 100 for the DISCOUNT zone
    re_curtailment = None
    if discount_zone:
        wind_col  = ZONE_WIND_COL.get(discount_zone)
        solar_col = ZONE_SOLAR_COL.get(discount_zone)
        load_col  = ZONE_LOAD_MAP.get(discount_zone)

        if load_col and load_col in df.columns:
            total_re = pd.Series(0.0, index=df.index)
            if wind_col and wind_col in df.columns:
                total_re = total_re + df[wind_col]
            if solar_col and solar_col in df.columns:
                total_re = total_re + df[solar_col]

            gross_load     = df[load_col].replace(0, pd.NA)
            re_curtailment = (total_re / gross_load * 100).mean()

    # ── Net Load Stress — as % of gross load on PREMIUM zone ─────────────
    # Net Load % = Net Load / Gross Load × 100
    net_load_stress_pct = None
    if premium_zone:
        nl_col   = ZONE_NET_LOAD_MAP.get(premium_zone)
        load_col = ZONE_LOAD_MAP.get(premium_zone)

        if (
            nl_col and nl_col in df.columns
            and load_col and load_col in df.columns
        ):
            gross_load          = df[load_col].replace(0, pd.NA)
            net_load_stress_pct = (df[nl_col] / gross_load * 100).mean()

    return {
        "avg_spread":      f"${avg_spread:.2f}",
        "volatility":      f"${volatility:.2f}" if pd.notna(volatility) else "N/A",
        "capture_rate":    f"{capture_rate:.1f}%",
        "re_curtailment":  f"{re_curtailment:.1f}%" if re_curtailment is not None else "N/A",
        "net_load_stress": f"{net_load_stress_pct:.1f}%" if net_load_stress_pct is not None else "N/A",
    }

