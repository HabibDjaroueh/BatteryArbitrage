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
    SPREAD_COLOR_MAP,
    _spread_label,
)
from src.tables import build_opportunity_table


def main() -> None:
    st.title("DAM Price Spreads")

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

    st.subheader("Spread Selection")

    overlay_col1, overlay_col2, overlay_col3 = st.columns([2, 2, 1])

    with overlay_col1:
        primary_spread = st.selectbox(
            "Primary Spread",
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
            "Overlay Spread (optional)",
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
        show_rolling = st.toggle("30D Rolling Mean", value=True)

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
    st.subheader("Market Snapshot")

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
        fig_ts = spread_time_series(filtered_df, selected_spreads, show_rolling)
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_right:
        fig_hist = spread_histogram(filtered_df, selected_spreads)
        st.plotly_chart(fig_hist, use_container_width=True)

    # 6. Regime tabs
    st.divider()
    st.subheader("Why Is This Spread Happening?")

    tab1, tab2, tab3 = st.tabs(
        [
            "📈  Net Load vs Spread",
            "🌬️  Renewables vs Spread",
            "📊  Net Load Views",
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
        fig_nl = net_load_vs_spread(
            filtered_df, spread_col, load_col, controls["region"]
        )
        st.plotly_chart(fig_nl, use_container_width=True)

    with tab2:
        st.caption(
            "How do wind and solar suppress or amplify the spread? "
            "Green markers = positive spread, red = negative."
        )
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
                "📅  Time Series",
                "🦆  Duck Curve",
            ]
        )

        with subtab1:
            st.caption(
                "Daily net load vs spread over the filtered window. "
                "Look for periods where net load spikes coincide "
                "with spread widening — these are the high-value dispatch windows."
            )
            fig_nl_ts = net_load_time_series(filtered_df, load_col, spread_col)
            st.plotly_chart(fig_nl_ts, use_container_width=True)

        with subtab2:
            st.caption(
                "Average net load by hour of day, split by year. "
                "The deepening midday trough shows solar growth. "
                "The steepening evening ramp is the primary battery "
                "arbitrage opportunity — charge in the trough, discharge on the ramp."
            )
            fig_duck = net_load_duck_curve(filtered_df, load_col)
            st.plotly_chart(fig_duck, use_container_width=True)

    # 7. Opportunity table + export
    st.divider()
    st.subheader("🎯 Spread Opportunities")
    st.caption(
        "Hourly observations ranked by absolute spread magnitude. "
        "Positive spread = selected zone is premium vs counterpart. "
        "Negative spread = selected zone is discount. "
        "Use this table to identify specific windows for battery dispatch."
    )

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
        st.caption(
            f"Showing top {len(opp_df_display)} of {len(opp_df)} total hours in filtered window  ·  "
            f"Avg spread: ${opp_df_display['Spread ($/MWh)'].mean():.2f}/MWh  ·  "
            f"Max spread: ${opp_df_display['Spread ($/MWh)'].abs().max():.2f}/MWh  ·  "
            f"Premium: {(opp_df_display['Direction'] == '▲ Premium').sum()} hrs  ·  "
            f"Discount: {(opp_df_display['Direction'] == '▼ Discount').sum()} hrs  ·  "
            f"Avg Wind: {opp_df_display['Wind Gen (MW)'].mean():,.0f} MW  ·  "
            f"Avg Net Load: {opp_df_display['Net Load (%)'].mean():.1f}%"
        )
    else:
        st.caption(f"No rows match the current filters. Total hours in window: {len(opp_df)}.")


if __name__ == "__main__":
    main()


