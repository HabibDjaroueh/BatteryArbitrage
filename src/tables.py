from __future__ import annotations

import pandas as pd

counterpart_map = {
    # Houston primary
    "spread_h_s": "LZ_SOUTH_DAM",
    "spread_h_w": "LZ_WEST_DAM",
    "spread_h_n": "LZ_NORTH_DAM",
    "spread_h_r": "LZ_RAYBN_DAM",
    # South primary
    "spread_s_h": "LZ_HOUSTON_DAM",
    "spread_s_w": "LZ_WEST_DAM",
    "spread_s_n": "LZ_NORTH_DAM",
    "spread_s_r": "LZ_RAYBN_DAM",
    # West primary
    "spread_w_h": "LZ_HOUSTON_DAM",
    "spread_w_s": "LZ_SOUTH_DAM",
    "spread_w_n": "LZ_NORTH_DAM",
    "spread_w_r": "LZ_RAYBN_DAM",
    # North primary
    "spread_n_h": "LZ_HOUSTON_DAM",
    "spread_n_s": "LZ_SOUTH_DAM",
    "spread_n_w": "LZ_WEST_DAM",
    "spread_n_r": "LZ_RAYBN_DAM",
    # East primary
    "spread_r_h": "LZ_HOUSTON_DAM",
    "spread_r_s": "LZ_SOUTH_DAM",
    "spread_r_w": "LZ_WEST_DAM",
    "spread_r_n": "LZ_NORTH_DAM",
}


def build_opportunity_table(
    df: pd.DataFrame,
    spread_col: str,
    load_col: str,
    region: str,
) -> pd.DataFrame:
    """
    Builds a ranked opportunity table from the filtered regional DataFrame.
    Each row is one hourly observation ranked by absolute spread magnitude.

    Parameters
    ----------
    df         : filtered regional DataFrame with datetime index
    spread_col : e.g. "spread_h_s"
    load_col   : e.g. "COAST_Load"
    region     : e.g. "coast" — used to resolve heat index and wind columns

    Returns
    -------
    pd.DataFrame with columns:
        Datetime, Spread ($/MWh), Direction, Hour,
        Wind Gen (MW), Net Load (%), Heat Index (°F)
    Sorted by absolute spread descending.
    """
    if df.empty or spread_col not in df.columns:
        return pd.DataFrame()

    # ── resolve region-specific columns ───────────────────────────────────
    REGION_NET_LOAD = {
        "coast": "COAST_Net_Load",
        "south": "SOUTH_Net_Load",
        "west": "WEST_Net_Load",
        "north": "NORTH_Net_Load",
        "east": "EAST_Net_Load",
    }
    REGION_HEAT_INDEX = {
        "coast": "COAST_Heat_Index",
        "south": "SOUTH_Heat_Index",
        "west": "WEST_Heat_Index",
        "north": "NORTH_Heat_Index",
        "east": "EAST_Heat_Index",
    }
    REGION_WIND = {
        "coast": "wind_gen_coast_mw",
        "south": "wind_gen_south_mw",
        "west": "wind_gen_west_mw",
        "north": "wind_gen_north_mw",
        "east": "wind_gen_east_mw",
    }

    net_load_col = REGION_NET_LOAD.get(region)
    heat_idx_col = REGION_HEAT_INDEX.get(region)
    wind_col = REGION_WIND.get(region)

    out = pd.DataFrame(index=df.index)

    # ── spread ─────────────────────────────────────────────────────────────
    out["Spread ($/MWh)"] = df[spread_col].round(2)

    # ── direction ──────────────────────────────────────────────────────────
    out["Direction"] = out["Spread ($/MWh)"].apply(
        lambda x: "▲ Premium" if x > 0 else ("▼ Discount" if x < 0 else "— Flat")
    )

    # ── hour ───────────────────────────────────────────────────────────────
    out["Hour"] = df.index.hour

    # ── wind generation ────────────────────────────────────────────────────
    if wind_col and wind_col in df.columns:
        out["Wind Gen (MW)"] = df[wind_col].round(1)
    else:
        out["Wind Gen (MW)"] = pd.Series(dtype="Float64", index=df.index)

    # ── net load % = net load / gross load × 100 ───────────────────────────
    if (
        net_load_col
        and net_load_col in df.columns
        and load_col
        and load_col in df.columns
    ):
        gross_load = df[load_col].replace(0, pd.NA)
        out["Net Load (%)"] = (df[net_load_col] / gross_load * 100).round(1)
    else:
        out["Net Load (%)"] = pd.Series(dtype="Float64", index=df.index)

    # ── heat index ─────────────────────────────────────────────────────────
    if heat_idx_col and heat_idx_col in df.columns:
        out["Heat Index (°F)"] = df[heat_idx_col].round(1)
    else:
        out["Heat Index (°F)"] = pd.Series(dtype="Float64", index=df.index)

    # ── sort by absolute spread descending ────────────────────────────────
    out["_abs_spread"] = out["Spread ($/MWh)"].abs()
    out = out.sort_values("_abs_spread", ascending=False).drop(columns=["_abs_spread"])

    # ── bring datetime in as a column ─────────────────────────────────────
    out.index.name = "Datetime"
    out = out.reset_index()
    out["Datetime"] = out["Datetime"].dt.strftime("%Y-%m-%d %H:%M")

    # ── final column order ─────────────────────────────────────────────────
    return out[
        [
            "Datetime",
            "Spread ($/MWh)",
            "Direction",
            "Hour",
            "Wind Gen (MW)",
            "Net Load (%)",
            "Heat Index (°F)",
        ]
    ]
