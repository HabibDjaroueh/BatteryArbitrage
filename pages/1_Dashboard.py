import streamlit as st

from src.data import load_region_df, get_lz_col, get_load_col
from src.controls import render_sidebar_controls, REGION_SPREADS
from src.filters import apply_filters
from src.kpis import (
    compute_kpis,
    premium_and_discount_zones,
    ZONE_LABEL,
    ZONE_LOAD_MAP,
    ZONE_WIND_COL,
    ZONE_SOLAR_COL,
    ZONE_NET_LOAD_MAP,
)
from src.charts import (
    spread_time_series,
    spread_histogram,
    net_load_vs_spread,
    renewables_vs_spread,
    net_load_time_series,
    net_load_duck_curve,
    monthly_spread_heatmap,
    monthly_summary_bars,
    day_of_week_spread,
    compute_monthly_stats,
    SPREAD_COLOR_MAP,
    _spread_label,
)
from src.tables import build_opportunity_table


def main() -> None:
    # Header bar (terminal style matching app.py)
    st.markdown(
        """
        <div style="background-color: #161b22; border-bottom: 1px solid #30363d; padding: 24px 8px; margin: -1rem -1rem 2rem -1rem; width: 100%;">
            <div>
                <div style="font-family: 'Courier New', monospace; font-size: 40px; font-weight: 700; color: #58a6ff; letter-spacing: 0.05em; margin: 4px 0 0 0; line-height: 1.1;">DAM Price Spreads</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Load + shared sidebar controls
    base_df = load_region_df("coast")
    controls = render_sidebar_controls(base_df)
    df = load_region_df(controls["region"])
    filtered_df = apply_filters(df, controls)

    # ── spread overlay selector ────────────────────────────────────────────────
    available_spreads = [
        c for c in REGION_SPREADS.get(controls["region"], [])
        if c in filtered_df.columns
    ]
    if not available_spreads:
        st.warning("No spread columns found for this region.")
        return

    # Control Panel (terminal style)
    st.markdown(
        '<div style="font-family: \'Courier New\', monospace; font-size: 1.1rem; color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem;">Control Panel</div>',
        unsafe_allow_html=True,
    )

    # Controls container
    with st.container(border=True):
        overlay_col1, overlay_col2, overlay_col3 = st.columns([2, 2, 1])

        with overlay_col1:
            primary_spread = st.selectbox(
                "Primary",
                options=available_spreads,
                index=(
                    available_spreads.index(controls["spread"])
                    if controls["spread"] and controls["spread"] in available_spreads
                    else 0
                ),
                format_func=lambda x: x.replace("spread_", "").replace("_", " → ").upper(),
                help="Primary spread — drives KPIs, regime tabs, and opportunity table"
            )

        with overlay_col2:
            overlay_options = ["None"] + [s for s in available_spreads if s != primary_spread]
            overlay_spread  = st.selectbox(
                "Overlay",
                options=overlay_options,
                index=0,
                format_func=lambda x: (
                    "— No Overlay —"
                    if x == "None"
                    else x.replace("spread_", "").replace("_", " → ").upper()
                ),
                help="Optional second spread overlaid on both charts in a contrasting color"
            )

        with overlay_col3:
            show_rolling = st.toggle("30D Roll", value=True)

    # Build selected spreads list — 1 or 2 entries
    selected_spreads = (
        [primary_spread]
        if overlay_spread == "None"
        else [primary_spread, overlay_spread]
    )

    # Primary spread drives all downstream components
    spread_col = primary_spread
    lz_col     = get_lz_col(controls["region"])
    load_col   = get_load_col(controls["region"])

    # 2. Market Snapshot — one row per selected spread
    st.markdown(
        '<div style="font-family: \'Courier New\', monospace; font-size: 1.1rem; color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem;">Market Snapshot</div>',
        unsafe_allow_html=True,
    )

    header_cols = st.columns([1.2, 1, 1, 1, 1, 1])
    header_cols[0].markdown("**Spread**")
    header_cols[1].markdown("**Avg Spread**")
    header_cols[2].markdown("**30D Volatility**")
    header_cols[3].markdown("**Capture Rate**")
    header_cols[4].markdown("**RE Pressure**")
    header_cols[5].markdown("**Net Load Stress**")

    st.divider()

    for i, sc in enumerate(selected_spreads):
        region_for_spread = next(
            (r for r, spreads in REGION_SPREADS.items() if sc in spreads),
            controls["region"],
        )
        zone_df       = load_region_df(region_for_spread)
        zone_filtered = apply_filters(zone_df, controls)

        premium_zone, discount_zone = premium_and_discount_zones(sc)
        # Ensure we have discount/premium zone columns for RE pressure and net load stress
        snapshot_df = zone_filtered.copy()
        for zone in (discount_zone, premium_zone):
            if zone is None or zone == region_for_spread:
                continue
            other_df = apply_filters(load_region_df(zone), controls)
            cols_to_add = [
                ZONE_WIND_COL.get(zone),
                ZONE_SOLAR_COL.get(zone),
                ZONE_LOAD_MAP.get(zone),
                ZONE_NET_LOAD_MAP.get(zone),
            ]
            for col in cols_to_add:
                if col and col in other_df.columns and col not in snapshot_df.columns:
                    snapshot_df[col] = other_df[col].reindex(snapshot_df.index)

        kpis   = compute_kpis(snapshot_df, sc)
        accent = SPREAD_COLOR_MAP.get(sc, "#94a3b8")
        label  = _spread_label(sc)

        premium_label  = ZONE_LABEL.get(premium_zone, "Premium Zone")
        discount_label = ZONE_LABEL.get(discount_zone, "Discount Zone")

        row_cols = st.columns([1.2, 1, 1, 1, 1, 1])

        row_cols[0].markdown(
            f"<span style='color:{accent}; font-size:18px;'>●</span> "
            f"<span style='font-weight:600; font-size:15px;'>{label}</span>",
            unsafe_allow_html=True,
        )
        row_cols[1].metric("", kpis["avg_spread"])
        row_cols[2].metric("", kpis["volatility"])
        row_cols[3].metric(
            "",
            kpis["capture_rate"],
            help=(
                f"Spread Capture Rate\n\n"
                f"= (hours where {label} > 0) / total hours × 100\n\n"
                f"How reliably the spread holds its direction. "
                f"50% = random. 65%+ = tradeable signal. 75%+ = strong."
            ),
        )
        row_cols[4].metric(
            "",
            kpis["re_curtailment"],
            help=(
                f"RE Curtailment Pressure — {discount_label}\n\n"
                f"= mean( (wind_gen_{discount_zone}_mw + solar_gen_{discount_zone}_mw) "
                f"/ {discount_zone.upper()}_Load × 100 )\n\n"
                f"How much of {discount_label}'s load is covered by renewables. "
                f"High % = renewables structurally suppressing {discount_label} prices."
            ),
        )
        row_cols[5].metric(
            "",
            kpis["net_load_stress"],
            help=(
                f"Net Load Stress — {premium_label}\n\n"
                f"= mean( {premium_zone.upper()}_Net_Load / {premium_zone.upper()}_Load × 100 )\n\n"
                f"How much of {premium_label}'s demand must be served by dispatchable generation. "
                f"High % = renewables covering little load, grid stressed, prices elevated."
            ),
        )

        if i < len(selected_spreads) - 1:
            st.markdown(
                "<hr style='border:none; border-top:1px solid #1e2d40; margin:4px 0;'>",
                unsafe_allow_html=True,
            )

    st.divider()

    # 4. Charts
    col_left, col_right = st.columns([1, 1])

    with col_left:
        with st.container(border=True):
            st.markdown(
                '<div style="font-family: \'Courier New\', monospace; font-size: 0.85rem; color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;">Spread Time Series</div>',
                unsafe_allow_html=True,
            )
            fig_ts = spread_time_series(filtered_df, selected_spreads, show_rolling)
            st.plotly_chart(fig_ts, use_container_width=True)

    with col_right:
        with st.container(border=True):
            st.markdown(
                '<div style="font-family: \'Courier New\', monospace; font-size: 0.85rem; color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;">Distribution</div>',
                unsafe_allow_html=True,
            )
            fig_hist = spread_histogram(filtered_df, selected_spreads)
            st.plotly_chart(fig_hist, use_container_width=True)

    # 6. Analysis Tabs
    st.divider()
    st.markdown(
        '<div style="font-family: \'Courier New\', monospace; font-size: 1.1rem; color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem;">Analysis</div>',
        unsafe_allow_html=True,
    )

    # Custom tab styling - terminal theme with blue accent (no red)
    st.markdown(
        """
        <style>
        /* Style Streamlit tabs to look like card buttons */
        div[data-baseweb="tabs"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 8px;
            margin-bottom: 1rem;
        }
        
        div[data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        button[data-baseweb="tab"] {
            font-family: 'Courier New', monospace !important;
            font-size: 0.9rem !important;
            background-color: #161b22 !important;
            color: #58a6ff !important;
            border: 1px solid #30363d !important;
            border-radius: 4px !important;
            padding: 8px 16px !important;
            transition: all 0.2s ease !important;
        }
        
        button[data-baseweb="tab"]:hover {
            background-color: #21262d !important;
            border-color: #58a6ff !important;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #58a6ff !important;
            color: #0a0e17 !important;
            border-color: #58a6ff !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #58a6ff !important;
        }
        
        button[data-baseweb="tab"][aria-selected="false"] {
            color: #58a6ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Analysis tabs with terminal styling
    analysis_tab1, analysis_tab2 = st.tabs(
        [
            "Monthly & Day Analysis",
            "Spread Regimes",
        ]
    )

    with analysis_tab1:
        st.caption(
            f"Monthly and day-of-week patterns in spread behavior. "
            "Use these views to identify seasonal trading windows "
            "and weekly scheduling patterns for battery dispatch."
        )
        
        spread_label = _spread_label(spread_col)
        lz_col = get_lz_col(controls["region"])
        
        # 1. Calendar Heatmap (full width)
        with st.container(border=True):
            st.markdown(
                '<div style="font-family:\'Courier New\',monospace; font-size:0.85rem; color:#58a6ff; margin-bottom:0.75rem;">CALENDAR HEATMAP</div>',
                unsafe_allow_html=True,
            )
            fig_heatmap = monthly_spread_heatmap(filtered_df, spread_col, "Spread", spread_label)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 2. Monthly bars (left) + Day of week (right)
        col_mon_l, col_mon_r = st.columns([3, 2])
        
        with col_mon_l:
            with st.container(border=True):
                fig_monthly = monthly_summary_bars(filtered_df, spread_col, "Spread", spread_label)
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col_mon_r:
            with st.container(border=True):
                fig_dow = day_of_week_spread(filtered_df, spread_col, "Spread", spread_label)
                st.plotly_chart(fig_dow, use_container_width=True)
        
        # 3. Monthly Statistics Table (full width)
        with st.container(border=True):
            st.markdown(
                '<div style="font-family:\'Courier New\',monospace; font-size:0.85rem; color:#58a6ff; margin-bottom:0.75rem;">MONTHLY STATISTICS</div>',
                unsafe_allow_html=True,
            )
            monthly_stats = compute_monthly_stats(filtered_df, spread_col, lz_col)
            
            if not monthly_stats.empty:
                st.dataframe(
                    monthly_stats,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Avg Spread ($/MWh)": st.column_config.NumberColumn(format="$%.2f"),
                        "Max Spread ($/MWh)": st.column_config.NumberColumn(format="$%.2f"),
                        "Min Spread ($/MWh)": st.column_config.NumberColumn(format="$%.2f"),
                        "Avg LZ Price ($/MWh)": st.column_config.NumberColumn(format="$%.2f"),
                        "Capture Rate (%)": st.column_config.ProgressColumn(
                            format="%.1f%%", min_value=0, max_value=100,
                        ),
                    },
                )
            else:
                st.info("No data available for the current filter selection.")

    with analysis_tab2:
        st.caption(
            "Regime analysis showing how different factors drive spread behavior. "
            "Use these views to understand the fundamental drivers of price spreads."
        )
        # Move the existing regime tabs content here
        tab1, tab2, tab3 = st.tabs(
            [
                "Net Load vs Spread",
                "Renewables vs Spread",
                "Net Load Views",
            ]
        )

        with tab1:
            st.caption(
                "How does regional load level drive the spread? "
                "A rising line means high load widens the spread. "
                "Load is reported in MW — ERCOT regional zones typically range "
                "from 5,000 MW (5 GW) off-peak to 25,000 MW (25 GW) at summer peak. "
                "The orange dotted line shows heat index (°F) — when both load "
                "and heat index are high simultaneously, spread spikes are most likely."
            )
            with st.container(border=True):
                fig_nl = net_load_vs_spread(
                    filtered_df, spread_col, load_col, controls["region"]
                )
                st.plotly_chart(fig_nl, use_container_width=True)

        with tab2:
            st.caption(
                "How do wind and solar suppress or amplify the spread? "
                "Green markers = positive spread, red = negative."
            )
            with st.container(border=True):
                fig_re = renewables_vs_spread(filtered_df, spread_col, controls["region"])
                st.plotly_chart(fig_re, use_container_width=True)

        with tab3:
            st.caption(
                "Three views of net load (regional load minus wind and solar). "
                "Net load is the most direct signal of grid stress — "
                "high net load means renewables aren't covering demand, "
                "prices rise and spreads widen. Use these charts to identify "
                "the net load thresholds where battery dispatch becomes valuable."
            )

            subtab1, subtab2 = st.tabs(
                [
                    "Time Series",
                    "Duck Curve",
                ]
            )

            with subtab1:
                st.caption(
                    "Daily net load vs spread over the filtered window. "
                    "Look for periods where net load spikes coincide "
                    "with spread widening — these are the high-value dispatch windows."
                )
                with st.container(border=True):
                    fig_nl_ts = net_load_time_series(filtered_df, load_col, spread_col)
                    st.plotly_chart(fig_nl_ts, use_container_width=True)

            with subtab2:
                st.caption(
                    "Average net load by hour of day, split by year. "
                    "The deepening midday trough shows solar growth. "
                    "The steepening evening ramp is the primary battery "
                    "arbitrage opportunity — charge in the trough, discharge on the ramp."
                )
                with st.container(border=True):
                    fig_duck = net_load_duck_curve(filtered_df, load_col)
                    st.plotly_chart(fig_duck, use_container_width=True)

    # 7. Opportunity table + export
    st.divider()
    st.markdown(
        '<div style="font-family: \'Courier New\', monospace; font-size: 1.1rem; color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem;">Spread Opportunities</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Hourly observations ranked by absolute spread magnitude. "
        "Positive spread = selected zone is premium vs counterpart. "
        "Negative spread = selected zone is discount. "
        "Use this table to identify specific windows for battery dispatch."
    )

    # Controls container
    with st.container(border=True):
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])

        with ctrl1:
            top_n = st.slider(
                "Top N Opportunities",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="Show the N hours with the largest absolute spread",
            )

        with ctrl2:
            direction_filter = st.selectbox(
                "Direction Filter",
                options=["All", "Premium Only (▲)", "Discount Only (▼)"],
                help="Filter by spread direction",
            )

        opp_df = build_opportunity_table(
            filtered_df, spread_col, load_col, controls["region"]
        )

        if direction_filter == "Premium Only (▲)":
            opp_df = opp_df[opp_df["Direction"] == "▲ Premium"]
        elif direction_filter == "Discount Only (▼)":
            opp_df = opp_df[opp_df["Direction"] == "▼ Discount"]

        opp_df_display = opp_df.head(top_n)

        with ctrl3:
            st.download_button(
                label="⬇️ Export CSV",
                data=opp_df_display.to_csv(index=False).encode("utf-8"),
                file_name=f"opportunities_{spread_col}_{controls['region']}.csv",
                mime="text/csv",
                help="Download the current table as a CSV file",
            )

    # Table container
    with st.container(border=True):
        st.dataframe(
            opp_df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Datetime": st.column_config.TextColumn(
                    "Datetime", width="medium"
                ),
                "Spread ($/MWh)": st.column_config.NumberColumn(
                    "Spread ($/MWh)",
                    format="$%.2f",
                    width="small"
                ),
                "Direction": st.column_config.TextColumn(
                    "Direction", width="small"
                ),
                "Hour": st.column_config.NumberColumn(
                    "Hour",
                    format="%d:00",
                    width="small",
                    help="Hour of day (0–23)"
                ),
                "Wind Gen (MW)": st.column_config.NumberColumn(
                    "Wind Gen (MW)",
                    format="%.1f MW",
                    width="small",
                    help="Regional wind generation at time of spread"
                ),
                "Net Load (%)": st.column_config.ProgressColumn(
                    "Net Load (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="medium",
                    help="Net load as % of gross load — high % means renewables covering little demand"
                ),
                "Heat Index (°F)": st.column_config.NumberColumn(
                    "Heat Index (°F)",
                    format="%.1f °F",
                    width="small",
                    help="Felt temperature accounting for humidity — driver of AC load"
                ),
            }
        )

    if len(opp_df_display) > 0:
        avg_spread = opp_df_display['Spread ($/MWh)'].mean()
        max_spread = opp_df_display['Spread ($/MWh)'].abs().max()
        st.markdown(
            f'<div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem;">'
            f'Showing top {len(opp_df_display)} of {len(opp_df)} total hours in filtered window  ·  '
            f'<span style="color: #58a6ff;">Avg spread: ${avg_spread:.2f}/MWh</span>  ·  '
            f'<span style="color: #58a6ff;">Max spread: ${max_spread:.2f}/MWh</span>  ·  '
            f'Premium: {(opp_df_display["Direction"] == "▲ Premium").sum()} hrs  ·  '
            f'Discount: {(opp_df_display["Direction"] == "▼ Discount").sum()} hrs  ·  '
            f'Avg Wind: {opp_df_display["Wind Gen (MW)"].mean():,.0f} MW  ·  '
            f'Avg Net Load: {opp_df_display["Net Load (%)"].mean():.1f}%'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"No rows match the current filters. Total hours in window: {len(opp_df)}.")


if __name__ == "__main__":
    main()
