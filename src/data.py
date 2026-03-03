from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import pandas as pd

try:
    import streamlit as st

    _cache_decorator: Callable = st.cache_data
except Exception:  # pragma: no cover - non-Streamlit environments
    def _cache_decorator(*args, **kwargs):  # type: ignore[override]
        def wrapper(func):
            return func

        return wrapper

REGIONAL_DATA_DIR = Path("data/regional_data")

REGION_FILES: Dict[str, str] = {
    "coast": "coast.csv",
    "south": "south.csv",
    "west": "west.csv",
    "north": "north.csv",
    "east": "east.csv",
    "master": "er_master.csv",
    "all": "all_regions.csv",
}


def _validate_region_name(region_name: str) -> None:
    if region_name not in REGION_FILES:
        valid = ", ".join(sorted(REGION_FILES))
        raise ValueError(f"Unknown region_name {region_name!r}. Expected one of: {valid}.")


@_cache_decorator(show_spinner=False)
def load_region_df(region_name: str) -> pd.DataFrame:
    """Load a single regional (or master) DataFrame with a datetime index and float columns."""
    _validate_region_name(region_name)

    csv_name = REGION_FILES[region_name]
    csv_path = REGIONAL_DATA_DIR / csv_name

    df = pd.read_csv(csv_path, parse_dates=["datetime"])

    if "datetime" not in df.columns:
        raise ValueError(f"'datetime' column not found in {csv_path}.")

    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Cast all feature columns to float, coercing errors to NaN
    feature_cols = df.columns
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    return df


@_cache_decorator(show_spinner=False)
def load_all_regions() -> Dict[str, pd.DataFrame]:
    """Load all five regional DataFrames as a dict keyed by region name."""
    regions = ["coast", "south", "west", "north", "east"]
    return {name: load_region_df(name) for name in regions}


def get_lz_col(region_name: str) -> str:
    """Return the LZ DAM price column for a given region key."""
    mapping = {
        "coast": "LZ_HOUSTON_DAM",
        "south": "LZ_SOUTH_DAM",
        "west": "LZ_WEST_DAM",
        "north": "LZ_NORTH_DAM",
        "east": "LZ_RAYBN_DAM",
    }
    try:
        return mapping[region_name]
    except KeyError:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown region_name {region_name!r} for LZ column. Expected one of: {valid}.")


def get_load_col(region_name: str) -> str:
    """Return the load column name for a given region key."""
    mapping = {
        "coast": "COAST_Load",
        "south": "SOUTH_Load",
        "west": "WEST_Load",
        "north": "NORTH_Load",
        "east": "EAST_Load",
    }
    try:
        return mapping[region_name]
    except KeyError:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown region_name {region_name!r} for load column. Expected one of: {valid}.")

