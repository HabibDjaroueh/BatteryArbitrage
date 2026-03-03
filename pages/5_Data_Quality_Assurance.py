import streamlit as st
import pandas as pd
from src.data import load_region_df
from src.qa import (
    date_coverage,
    missing_values_summary,
    duplicate_timestamps,
)

st.title("Data Quality Assurance")
st.caption(
    "Data quality report across all regional DataFrames. "
    "Use this page to verify coverage, identify gaps, and "
    "confirm data integrity before drawing conclusions."
)

# ── sidebar — region only ──────────────────────────────────────────────────
st.sidebar.title("Filters")
region = st.sidebar.selectbox(
    "Region",
    options=["coast", "south", "west", "north", "east"],
    format_func=lambda x: x.capitalize(),
    help="Select the ERCOT region to inspect"
)

df = load_region_df(region)

# ── section 1: date coverage ───────────────────────────────────────────────
st.divider()
st.subheader("📅 Date Coverage — All Regions")
st.caption(
    "Checks that each regional DataFrame covers the expected hourly "
    "range with no major gaps. Coverage is capped at 100% — values "
    "slightly above 100% before capping indicate DST duplicate rows "
    "which are automatically removed at load time."
)

try:
    coverage_df = date_coverage(df)
    st.dataframe(
        coverage_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Coverage (%)": st.column_config.ProgressColumn(
                "Coverage (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
            "Status": st.column_config.TextColumn(
                "Status", width="small"
            )
        }
    )
except Exception as e:
    st.error(f"Date coverage check failed: {e}")

# ── section 2: missing values ──────────────────────────────────────────────
st.divider()
st.subheader(f"🔍 Missing Values — {region.capitalize()}")
st.caption(
    "Per-column null count for the selected region. "
    "✅ = no nulls · ⚠️ = < 5% missing · ❌ = > 5% missing"
)

try:
    missing_df    = missing_values_summary(df, region)
    total_missing = missing_df["Missing"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Columns",       len(missing_df))
    col2.metric("Columns with Nulls",  (missing_df["Missing"] > 0).sum())
    col3.metric("Total Missing Values", f"{total_missing:,}")

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Missing (%)": st.column_config.ProgressColumn(
                "Missing (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
            "Status": st.column_config.TextColumn(
                "Status", width="small"
            )
        }
    )
except Exception as e:
    st.error(f"Missing values check failed: {e}")

# ── section 3: duplicate timestamps ───────────────────────────────────────
st.divider()
st.subheader(f"🔁 Duplicate Timestamps — {region.capitalize()}")
st.caption(
    "Duplicate datetime index entries corrupt time series analysis — "
    "resampling double-counts affected hours and lagged features misalign. "
    "In ERCOT data, duplicates arise from Daylight Saving Time fallback: "
    "clocks roll back on the first Sunday of November and 02:00 appears "
    "twice in the raw data. "
    "This pipeline resolves DST duplicates at load time in src/data.py "
    "by retaining the first occurrence and discarding the second. "
    "The check below confirms clean data post-load."
)

try:
    dupes_df = duplicate_timestamps(df, region)
    if len(dupes_df) == 0:
        st.success(
            f"✅ No duplicate timestamps found in "
            f"{region.capitalize()} data."
        )
    else:
        st.warning(f"⚠️ {len(dupes_df)} duplicate timestamp rows found.")
        st.dataframe(dupes_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Duplicate check failed: {e}")
