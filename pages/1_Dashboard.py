import pandas as pd
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
    detect_anomalous_days,
    anomaly_scatter_chart,
    build_anomaly_fundamentals,
    anomaly_fundamentals_radar,
    find_similar_days,
    similar_days_spread_distribution,
    compute_regime_signal,
    zone_from_column,
    ZONE_TEMP_COL,
    ZONE_HUMIDITY_COL,
    ZONE_HEAT_INDEX_COL,
    LZ_PRICE_COLS,
    SPREAD_COLOR_MAP,
    _spread_label,
)
from src.tables import build_opportunity_table


def _card_label(label: str) -> None:
    """Render a small card label inside a chart container."""
    st.markdown(
        f'<div style="font-family: \'Courier New\', monospace; font-size: 0.85rem; '
        f'color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; '
        f'margin-bottom: 0.75rem;">{label}</div>',
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, color: str = "#58a6ff", sub: str = "") -> None:
    """Render a styled KPI card with colored value."""
    sub_html = f'<div style="font-size:0.7rem; color:#8b949e; margin-top:2px;">{sub}</div>' if sub else ""
    st.markdown(
        f'<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; '
        f'padding:12px 16px; text-align:center;">'
        f'<div style="font-family:\'Courier New\',monospace; font-size:0.7rem; '
        f'color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">{label}</div>'
        f'<div style="font-family:\'Courier New\',monospace; font-size:1.4rem; '
        f'font-weight:700; color:{color};">{value}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _regime_badge(regime: str, strength: int, vol_regime: str) -> None:
    """
    Render a regime indicator badge showing Bullish/Bearish/Neutral status.
    
    Parameters
    ----------
    regime : 'Bullish' / 'Bearish' / 'Neutral' / 'N/A'
    strength : Signal strength (0-100)
    vol_regime : 'High Vol' / 'Low Vol' / 'Normal' / 'N/A'
    """
    # Color scheme adapted for terminal theme (blue/grey, no red/green)
    regime_colors = {
        "Bullish": ("#58a6ff", "#0d2847"),      # Blue text, dark blue bg
        "Bearish": ("#7ab8ff", "#1a3a5c"),     # Lighter blue text, darker blue bg
        "Neutral": ("#94a3b8", "#1e293b"),      # Gray text, dark gray bg
        "N/A": ("#475569", "#1e293b"),          # Muted gray
    }
    
    # Volatility regime colors (terminal theme)
    vol_colors = {
        "High Vol": "#7ab8ff",   # Lighter blue
        "Low Vol": "#58a6ff",    # Blue
        "Normal": "#94a3b8",     # Gray
        "N/A": "#475569",        # Muted gray
    }
    
    # Get colors for current regime
    text_color, bg_color = regime_colors.get(regime, ("#94a3b8", "#1e293b"))
    vol_color = vol_colors.get(vol_regime, "#94a3b8")

    # Render badge HTML
    st.markdown(
        f'<div style="background:{bg_color}; border:1px solid {text_color}40; border-radius:8px; '
        f'padding:16px; text-align:center;">'
        f'<div style="font-family:\'Courier New\',monospace; font-size:0.7rem; '
        f'color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">Regime</div>'
        f'<div style="font-family:\'Courier New\',monospace; font-size:1.6rem; '
        f'font-weight:700; color:{text_color}; margin:4px 0;">{regime}</div>'
        f'<div style="display:flex; justify-content:center; gap:12px; margin-top:6px;">'
        f'<span style="font-size:0.75rem; color:{vol_color};">{vol_regime}</span>'
        f'<span style="font-size:0.75rem; color:#8b949e;">Strength: {strength}/100</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


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

    header_cols = st.columns([1.3, 1.2, 1, 1, 1, 1, 1])
    header_cols[0].markdown("**Regime**")
    header_cols[1].markdown("**Spread**")
    header_cols[2].markdown("**Avg Spread**")
    header_cols[3].markdown("**30D Volatility**")
    header_cols[4].markdown("**Capture Rate**")
    header_cols[5].markdown("**RE Pressure (D)**")
    header_cols[6].markdown("**Net Load Stress (P)**")

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

        # Calculate regime signal
        regime = compute_regime_signal(snapshot_df, sc)

        row_cols = st.columns([1.3, 1.2, 1, 1, 1, 1, 1])

        # Regime badge (first column)
        with row_cols[0]:
            _regime_badge(regime["regime"], regime["signal_strength"], regime["vol_regime"])

        # Spread label (second column)
        row_cols[1].markdown(
            f"<span style='color:{accent}; font-size:18px;'>●</span> "
            f"<span style='font-weight:600; font-size:15px;'>{label}</span>",
            unsafe_allow_html=True,
        )
        row_cols[2].metric("", kpis["avg_spread"])
        row_cols[3].metric("", kpis["volatility"])
        row_cols[4].metric(
            "",
            kpis["capture_rate"],
            help=(
                f"Spread Capture Rate\n\n"
                f"= (hours where {label} > 0) / total hours × 100\n\n"
                f"How reliably the spread holds its direction. "
                f"50% = random. 65%+ = tradeable signal. 75%+ = strong."
            ),
        )
        row_cols[5].metric(
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
        row_cols[6].metric(
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
    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(
        [
            "Monthly & Day Analysis",
            "Anomalous Days",
            "Similar Days Lookup",
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
            f"Identify days where spread behavior was statistically unusual. "
            "Examine the weather and grid conditions that drove these anomalies "
            "to build intuition for when extreme spreads are likely."
        )
        
        # Anomaly controls
        with st.container(border=True):
            anom_c1, anom_c2, anom_c3 = st.columns([2, 1, 1])
            
            with anom_c1:
                # Column selection dropdown
                anom_options = list(available_spreads)
                if controls["region"] == "all":
                    hub_cols = [c for c in LZ_PRICE_COLS.keys() if c in filtered_df.columns]
                    anom_options = hub_cols + anom_options
                
                _lz_labels = {
                    "LZ_HOUSTON_DAM": "Houston Hub",
                    "LZ_SOUTH_DAM": "South Hub", 
                    "LZ_WEST_DAM": "West Hub",
                    "LZ_NORTH_DAM": "North Hub",
                    "LZ_RAYBN_DAM": "Rayburn Hub",
                }
                
                def _anom_label(x):
                    if x in _lz_labels:
                        return _lz_labels[x]
                    return _spread_label(x)
                
                anom_col = st.selectbox(
                    "Column to Analyze",
                    options=anom_options,
                    index=anom_options.index(spread_col) if spread_col in anom_options else 0,
                    format_func=_anom_label,
                    key="anom_spread_select",
                )
            
            with anom_c2:
                z_thresh = st.slider(
                    "Z-Score Threshold",
                    min_value=1.0, max_value=4.0, value=2.0, step=0.25,
                    help="Standard deviations from mean. Lower = more anomalies detected.",
                    key="z_thresh_slider",
                )
            
            with anom_c3:
                anom_direction = st.selectbox(
                    "Direction",
                    options=["Both", "Spikes Only", "Crashes Only"],
                    key="anom_direction",
                )
        
        # Detect anomalies
        anomalies = detect_anomalous_days(filtered_df, anom_col, z_thresh)
        
        # Filter by direction
        if anom_direction == "Spikes Only" and not anomalies.empty:
            anomalies = anomalies[anomalies["direction"] == "Spike"]
        elif anom_direction == "Crashes Only" and not anomalies.empty:
            anomalies = anomalies[anomalies["direction"] == "Crash"]
        
        # Handle no anomalies case
        if anomalies.empty:
            st.info(
                f"No anomalous days found at z ≥ {z_thresh}. "
                "Try lowering the threshold or expanding the date range."
            )
        else:
            # Summary KPIs
            n_spikes = (anomalies["direction"] == "Spike").sum()
            n_crashes = (anomalies["direction"] == "Crash").sum()
            max_z = anomalies["z_score"].abs().max()
            
            s1, s2, s3, s4 = st.columns(4)
            
            with s1:
                st.markdown(
                    f'<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; '
                    f'padding:12px 16px; text-align:center;">'
                    f'<div style="font-family:\'Courier New\',monospace; font-size:0.7rem; '
                    f'color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">Anomalous Days</div>'
                    f'<div style="font-family:\'Courier New\',monospace; font-size:1.4rem; '
                    f'font-weight:700; color:#58a6ff;">{len(anomalies)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            
            with s2:
                st.markdown(
                    f'<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; '
                    f'padding:12px 16px; text-align:center;">'
                    f'<div style="font-family:\'Courier New\',monospace; font-size:0.7rem; '
                    f'color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">Spikes</div>'
                    f'<div style="font-family:\'Courier New\',monospace; font-size:1.4rem; '
                    f'font-weight:700; color:#58a6ff;">{n_spikes}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            
            with s3:
                st.markdown(
                    f'<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; '
                    f'padding:12px 16px; text-align:center;">'
                    f'<div style="font-family:\'Courier New\',monospace; font-size:0.7rem; '
                    f'color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">Crashes</div>'
                    f'<div style="font-family:\'Courier New\',monospace; font-size:1.4rem; '
                    f'font-weight:700; color:#7ab8ff;">{n_crashes}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            
            with s4:
                st.markdown(
                    f'<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; '
                    f'padding:12px 16px; text-align:center;">'
                    f'<div style="font-family:\'Courier New\',monospace; font-size:0.7rem; '
                    f'color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">Max |z|</div>'
                    f'<div style="font-family:\'Courier New\',monospace; font-size:1.4rem; '
                    f'font-weight:700; color:#58a6ff;">{max_z:.1f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            
            # Scatter chart
            with st.container(border=True):
                st.markdown(
                    '<div style="font-family:\'Courier New\',monospace; font-size:0.85rem; color:#58a6ff; margin-bottom:0.75rem;">ANOMALOUS DAYS TIMELINE</div>',
                    unsafe_allow_html=True,
                )
                spread_label_anom = _spread_label(anom_col) if anom_col.startswith("spread_") else None
                fig_anom = anomaly_scatter_chart(filtered_df, anom_col, anomalies, "Spread", spread_label_anom)
                st.plotly_chart(fig_anom, use_container_width=True)
            
            # Fundamentals table
            fund_df = build_anomaly_fundamentals(
                filtered_df, anomalies.index, controls["region"], anom_col=anom_col
            )
            
            if not fund_df.empty:
                # Merge anomalies with fundamentals
                display_df = anomalies.reset_index()
                val_label = "Avg Spread ($/MWh)"
                display_df.columns = ["Date", val_label, "Z-Score", "Direction"]
                display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
                merged = display_df.merge(fund_df, on="Date", how="left")
                
                with st.container(border=True):
                    st.markdown(
                        '<div style="font-family:\'Courier New\',monospace; font-size:0.85rem; color:#58a6ff; margin-bottom:0.75rem;">ANOMALOUS DAY FUNDAMENTALS</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        merged.head(30),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            val_label: st.column_config.NumberColumn(format="$%.2f"),
                            "Z-Score": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )
                
                # Radar chart deep dive
                if len(merged) > 0:
                    with st.container(border=True):
                        st.markdown(
                            '<div style="font-family:\'Courier New\',monospace; font-size:0.85rem; color:#58a6ff; margin-bottom:0.75rem;">DEEP DIVE — COMPARE ANOMALOUS DAY VS BASELINE</div>',
                            unsafe_allow_html=True,
                        )
                        selected_day = st.selectbox(
                            "Select Anomalous Day",
                            options=merged["Date"].tolist(),
                            key="radar_day_select",
                        )
                        
                        # Prepare data for radar chart
                        row = merged[merged["Date"] == selected_day].iloc[0]
                        fundamental_labels = [c for c in merged.columns 
                                           if c not in ("Date", val_label, "Z-Score", "Direction")]
                        
                        anom_row = {}
                        for lab in fundamental_labels:
                            val = row.get(lab)
                            if pd.notna(val):
                                anom_row[lab] = float(val)
                        
                        # Calculate baseline
                        effective_region = controls["region"]
                        if controls["region"] == "all":
                            effective_region = zone_from_column(anom_col) or "coast"
                        
                        baseline = {}
                        col_map = {
                            "Temp (°C)": ZONE_TEMP_COL.get(effective_region),
                            "Humidity (%)": ZONE_HUMIDITY_COL.get(effective_region),
                            "Heat Index (°F)": ZONE_HEAT_INDEX_COL.get(effective_region),
                            "Wind Gen (MW)": ZONE_WIND_COL.get(effective_region),
                            "Solar Gen (MW)": ZONE_SOLAR_COL.get(effective_region),
                            "Load (MW)": ZONE_LOAD_MAP.get(effective_region),
                            "Net Load (MW)": ZONE_NET_LOAD_MAP.get(effective_region),
                        }
                        
                        for lab in fundamental_labels:
                            col = col_map.get(lab)
                            if col and col in filtered_df.columns:
                                baseline[lab] = float(filtered_df[col].mean())
                        
                        if anom_row and baseline:
                            fig_radar = anomaly_fundamentals_radar(anom_row, baseline, selected_day)
                            st.plotly_chart(fig_radar, use_container_width=True)
                            st.caption(
                                "Values shown as % of baseline period average. "
                                "100% = at baseline. >100% = above average."
                            )

    with analysis_tab3:
        st.caption(
            "Planning for a future day? Enter expected conditions below and find "
            "the most similar historical days. Their spread outcomes give you a "
            "data-driven prior for what to expect."
        )

        # Build available fundamental inputs
        # When region="all", default to Houston zone for fundamentals
        _sim_region = controls["region"]
        if controls["region"] == "all":
            _sim_region = "coast"
        
        available_inputs = {}
        for label, col_map_dict in [
            ("Temp (°C)", ZONE_TEMP_COL),
            ("Humidity (%)", ZONE_HUMIDITY_COL),
            ("Heat Index (°F)", ZONE_HEAT_INDEX_COL),
            ("Wind Gen (MW)", ZONE_WIND_COL),
            ("Solar Gen (MW)", ZONE_SOLAR_COL),
            ("Load (MW)", ZONE_LOAD_MAP),
        ]:
            col = col_map_dict.get(_sim_region)
            if col and col in filtered_df.columns:
                series = filtered_df[col].dropna()
                if not series.empty:
                    available_inputs[label] = {
                        "col": col,
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "p25": float(series.quantile(0.25)),
                        "p75": float(series.quantile(0.75)),
                    }

        if not available_inputs:
            st.info("No fundamental columns available for this region.")
        else:
            with st.container(border=True):
                _card_label("Target Day Conditions")
                st.markdown(
                    '<div style="font-size:0.8rem; color:#8b949e; margin-bottom:12px;">'
                    'Set your expected conditions. Defaults are period averages. '
                    'Sliders show the historical range.</div>',
                    unsafe_allow_html=True,
                )

                # Create input sliders in columns
                target_values = {}
                n_inputs = len(available_inputs)
                cols_per_row = 3
                labels = list(available_inputs.keys())

                for row_start in range(0, n_inputs, cols_per_row):
                    row_labels = labels[row_start:row_start + cols_per_row]
                    cols = st.columns(len(row_labels))
                    for col_ui, lab in zip(cols, row_labels):
                        info = available_inputs[lab]
                        with col_ui:
                            target_values[lab] = st.slider(
                                lab,
                                min_value=info["min"],
                                max_value=info["max"],
                                value=info["mean"],
                                key=f"sim_{lab}",
                                help=f"Historical range: {info['min']:.1f} – {info['max']:.1f}",
                            )

                sim_n = st.slider(
                    "Number of Similar Days",
                    min_value=5, max_value=50, value=15, step=5,
                    key="sim_n_days",
                )

            # Find similar days
            if target_values:
                similar_df = find_similar_days(
                    filtered_df, target_values, controls["region"], spread_col, n=sim_n
                )

                if similar_df.empty:
                    st.info("Not enough historical data to find similar days.")
                else:
                    # Determine outcome column name
                    outcome_col = "Avg Spread ($/MWh)" if "Avg Spread ($/MWh)" in similar_df.columns else "Avg Price ($/MWh)"
                    if outcome_col not in similar_df.columns:
                        # Fallback — find the first matching column
                        outcome_col = next(
                            (c for c in similar_df.columns if "Avg" in c and "$/MWh" in c),
                            None,
                        )

                    if outcome_col and outcome_col in similar_df.columns:
                        values = similar_df[outcome_col]
                        sm1, sm2, sm3, sm4 = st.columns(4)
                        with sm1:
                            _kpi_card(f"Expected Spread", f"${values.mean():.2f}", "#58a6ff",
                                      f"Mean of {len(similar_df)} similar days")
                        with sm2:
                            _kpi_card("Upside (P75)", f"${values.quantile(0.75):.2f}", "#7ab8ff")  # Lighter blue
                        with sm3:
                            _kpi_card("Downside (P25)", f"${values.quantile(0.25):.2f}", "#94a3b8")  # Grey
                        with sm4:
                            pct_positive = (values > 0).mean() * 100
                            _kpi_card("% Positive", f"{pct_positive:.0f}%", "#58a6ff")

                    # Distribution chart + table side by side
                    col_sim_l, col_sim_r = st.columns([2, 3])

                    with col_sim_l:
                        with st.container(border=True):
                            fig_sim_dist = similar_days_spread_distribution(
                                similar_df, col_label="Spread",
                            )
                            st.plotly_chart(fig_sim_dist, use_container_width=True)

                    with col_sim_r:
                        with st.container(border=True):
                            _card_label("Similar Days Detail")
                            st.dataframe(
                                similar_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Distance": st.column_config.NumberColumn(format="%.3f"),
                                },
                            )

    with analysis_tab4:
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
