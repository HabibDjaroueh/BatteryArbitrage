import streamlit as st
import pandas as pd
from src.data import load_region_df
from src.qa import (
    date_coverage,
    missing_values_summary,
    duplicate_timestamps,
)


def _section_header(label: str) -> None:
    st.markdown(
        f"<div style=\"font-family: 'Courier New', monospace; font-size: 1.1rem; "
        f"color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; "
        f"margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; "
        f"padding-bottom: 0.5rem;\">{label}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    # Header bar (terminal style)
    st.markdown(
        """
        <div style="background-color: #161b22; border-bottom: 1px solid #30363d;
                    padding: 24px 8px; margin: -1rem -1rem 2rem -1rem;
                    display: flex; justify-content: space-between;
                    align-items: center; width: 100%;">
            <div>
                <div style="font-family: 'Courier New', monospace; font-size: 40px;
                            font-weight: 700; color: #58a6ff;
                            letter-spacing: 0.05em; margin: 4px 0 0 0;
                            line-height: 1.1;">
                    Data Quality Assurance
                </div>
            </div>
            <div style="font-family: 'Courier New', monospace; font-size: 16px;
                        color: #8b949e; opacity: 0.6; text-align: right;">
                &nbsp;
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Checks coverage, missing values and duplicate timestamps so you can trust downstream analytics."
    )

    # Sidebar — region only
    st.sidebar.markdown("### Filters")
    region = st.sidebar.selectbox(
        "Region",
        options=["coast", "south", "west", "north", "east"],
        format_func=lambda x: x.capitalize(),
        help="Select the ERCOT region to inspect",
    )

    df = load_region_df(region)

    # 1. Date coverage
    st.divider()
    _section_header("Date Coverage")
    st.caption(
        "Verifies hourly coverage for each region. Coverage above 100% before capping is due to DST duplicates removed at load time."
    )

    try:
        coverage_df = date_coverage(df)
        with st.container(border=True):
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
                    ),
                },
            )
    except Exception as e:
        st.error(f"Date coverage check failed: {e}")

    # 2. Missing values
    st.divider()
    _section_header(f"Missing Values — {region.capitalize()}")
    st.caption(
        "Per-column null counts for the selected region; status reflects null share thresholds."
    )

    try:
        missing_df = missing_values_summary(df, region)
        total_missing = missing_df["Missing"].sum()

        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Columns", len(missing_df))
            col2.metric("Columns with Nulls", (missing_df["Missing"] > 0).sum())
            col3.metric("Total Missing Values", f"{total_missing:,}")

        with st.container(border=True):
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
                    ),
                },
            )
    except Exception as e:
        st.error(f"Missing values check failed: {e}")

    # 3. Duplicate timestamps
    st.divider()
    _section_header(f"Duplicate Timestamps — {region.capitalize()}")
    st.caption(
        "Detects duplicate datetime index entries. DST fallback in November can create raw duplicates; the loader keeps the first hour and drops the second."
    )

    try:
        dupes_df = duplicate_timestamps(df, region)
        with st.container(border=True):
            if len(dupes_df) == 0:
                st.info(
                    f"No duplicate timestamps found in {region.capitalize()} data."
                )
            else:
                st.warning(
                    f"{len(dupes_df)} duplicate timestamp rows found; review before using this region."
                )
                st.dataframe(dupes_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Duplicate check failed: {e}")


if __name__ == "__main__":
    main()
