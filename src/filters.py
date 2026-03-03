from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _get_season_months(season: str) -> List[int]:
    season = season.capitalize()
    mapping = {
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
        "Winter": [12, 1, 2],
        "All": [],
    }
    return mapping.get(season, [])


def apply_filters(df: pd.DataFrame, controls: Dict[str, Any]) -> pd.DataFrame:
    """Apply date, hour, day-type, and season filters to df based on controls."""
    if df.empty:
        return df

    filtered = df

    # 1. Date range
    date_range = controls.get("date_range", (None, None))
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 2:
            start, end = date_range
            if start is not None and end is not None:
                filtered = filtered.loc[str(start) : str(end)]
        elif len(date_range) == 1:
            start = date_range[0]
            if start is not None:
                filtered = filtered.loc[str(start) :]

    # 2. Hour range
    if isinstance(filtered.index, pd.DatetimeIndex):
        low, high = controls.get("hour_range", (0, 23))
        hours = filtered.index.hour
        mask = (hours >= low) & (hours <= high)
        filtered = filtered[mask]

    # 3. Day type
    day_type = controls.get("day_type", "All")
    if day_type == "Weekday":
        if {"weekend", "holiday"}.issubset(filtered.columns):
            filtered = filtered[(filtered["weekend"] == 0) & (filtered["holiday"] == 0)]
    elif day_type == "Weekend":
        if "weekend" in filtered.columns:
            filtered = filtered[filtered["weekend"] == 1]
    elif day_type == "Holiday":
        if "holiday" in filtered.columns:
            filtered = filtered[filtered["holiday"] == 1]

    # 4. Season
    season = controls.get("season", "All")
    months = _get_season_months(season)
    if months and isinstance(filtered.index, pd.DatetimeIndex):
        filtered = filtered[filtered.index.month.isin(months)]

    # 5. Guard rails
    if filtered.empty:
        try:
            import streamlit as st

            st.warning(
                "No data matches the selected filters — showing full dataset."
            )
        except Exception:
            # If Streamlit isn't available, just fall back silently
            pass
        return df

    return filtered

