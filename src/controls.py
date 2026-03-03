from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

REGION_SPREADS = {
    "coast": ["spread_h_s", "spread_h_w", "spread_h_n", "spread_h_r"],
    "south": ["spread_s_h", "spread_s_w", "spread_s_n", "spread_s_r"],
    "west":  ["spread_w_h", "spread_w_s", "spread_w_n", "spread_w_r"],
    "north": ["spread_n_h", "spread_n_s", "spread_n_w", "spread_n_r"],
    "east":  ["spread_r_h", "spread_r_s", "spread_r_w", "spread_r_n"],
}

# Flat list of all spreads — used for leakage exclusion in models.py only
AVAILABLE_SPREADS = [s for spreads in REGION_SPREADS.values() for s in spreads]


def render_sidebar_controls(df: pd.DataFrame) -> Dict[str, Any]:
    """Render shared sidebar controls and return a controls dict."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-content">
                <div class="sidebar-title" style="margin-bottom: 1.2rem;">Filters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 1. Region selector — MUST be first ────────────────────────────────
    region = st.sidebar.selectbox(
        "Region",
        options=["coast", "south", "west", "north", "east"],
        format_func=lambda x: x.capitalize(),
        help="Select the ERCOT region",
    )

    # ── 2. Spread selector — options driven by selected region ─────────────
    region_spread_cols = REGION_SPREADS.get(region, [])
    if not region_spread_cols:
        st.sidebar.warning("No spread options for this region.")
        spread = None
    else:
        spread = st.sidebar.selectbox(
            "Spread",
            options=region_spread_cols,
        format_func=lambda x: (
            x.replace("spread_", "")
             .replace("_", " → ")
             .upper()
        ),
            help="Spread options update automatically based on selected region",
        )

    # ── 3. Remaining filters ──────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for date filtering.")

    min_date = df.index.min().date()
    max_date = df.index.max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Ensure we always have a complete (start, end) pair before returning
    if not isinstance(date_range, (list, tuple)) or len(date_range) < 2:
        date_range = (min_date, max_date)

    hour_range = st.sidebar.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=(0, 23),
    )

    day_type = st.sidebar.selectbox(
        "Day Type",
        options=["All", "Weekday", "Weekend", "Holiday"],
    )

    season = st.sidebar.selectbox(
        "Season",
        options=["All", "Spring", "Summer", "Autumn", "Winter"],
    )

    return {
        "region": region,
        "spread": spread,
        "date_range": date_range,
        "hour_range": hour_range,
        "day_type": day_type,
        "season": season,
    }

