import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data import load_region_df
from src.controls import REGION_SPREADS
from src.models import train_model, generate_summary, LZ_COLS


def main():

    # Header bar (terminal style matching dashboard/app)
    st.markdown(
        """
        <div style="background-color: #161b22; border-bottom: 1px solid #30363d; padding: 24px 8px; margin: -1rem -1rem 2rem -1rem; display: flex; justify-content: space-between; align-items: center; width: 100%;">
            <div>
                <div style="font-family: 'Courier New', monospace; font-size: 40px; font-weight: 700; color: #58a6ff; letter-spacing: 0.05em; margin: 4px 0 0 0; line-height: 1.1;">Spread Drivers</div>
            </div>
            <div style="font-family: 'Courier New', monospace; font-size: 16px; color: #8b949e; opacity: 0.6; text-align: right;">&nbsp;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Button styling – blue terminal theme
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #1f6feb !important;
            color: #ffffff !important;
            border: 1px solid #58a6ff !important;
            border-radius: 4px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 0.9rem !important;
        }
        div.stButton > button:hover {
            background-color: #2563eb !important;
            border-color: #58a6ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Train an XGBoost model on the filtered dataset to identify "
        "which fundamentals drive your selected spread or LZ price. "
        "Use the sidebar to select a region and apply filters before training."
    )

    # ── sidebar: region only ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-content">
                <div class="sidebar-title" style="margin-bottom: 1.2rem;">Filters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    region = st.sidebar.selectbox(
        "Region",
        options=["coast", "south", "west", "north", "east"],
        format_func=lambda x: x.capitalize(),
        help="Select the ERCOT region to train on"
    )
    df = load_region_df(region)

    st.divider()

    # ── model configuration / control panel ─────────────────────────────────
    st.markdown(
        '<div style="font-family: \'Courier New\', monospace; font-size: 1.1rem; '
        'color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; '
        'margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; '
        'padding-bottom: 0.5rem;">Control Panel</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        # ── Two column layout: dropdowns left, sliders right ────────────────
        left_col, right_col = st.columns(2)

        # ── LEFT — stacked dropdowns ────────────────────────────────────────
        with left_col:
            available_spreads = [
                c for c in REGION_SPREADS.get(region, []) if c in df.columns
            ]
            available_lz = [c for c in LZ_COLS if c in df.columns]
            all_targets = available_spreads + available_lz

            if not all_targets:
                st.error("No valid target columns found.")
                return

            target_col = st.selectbox(
                "Prediction Target",
                options=all_targets,
                index=0,
                format_func=lambda x: (
                    "Spread: "
                    + x.replace("spread_", "").replace("_", " → ").upper()
                    if "spread" in x
                    else "LZ Price: " + x.replace("_DAM", "").replace("_", " ")
                ),
                help="The variable the model will learn to predict",
            )

            day_type = st.selectbox(
                "Day Type",
                options=["All", "Weekday", "Weekend", "Holiday"],
                help="Restrict training to specific day types",
            )

            season = st.selectbox(
                "Season",
                options=["All", "Spring", "Summer", "Autumn", "Winter"],
                help="Restrict training to a specific season",
            )

        # ── RIGHT — stacked sliders ─────────────────────────────────────────
        with right_col:
            min_date = df.index.min().date()
            max_date = df.index.max().date()

            date_range = st.date_input(
                "Training Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Select the date window to train on",
            )
            if not isinstance(date_range, (list, tuple)) or len(date_range) < 2:
                date_range = (min_date, max_date)

            hour_range = st.slider(
                "Hours of Day",
                min_value=0,
                max_value=23,
                value=(0, 23),
                help="Restrict training to specific hours of the day",
            )

            top_n_features = st.slider(
                "Top N Features",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of features to display in the importance chart",
            )

    st.divider()

    # ── apply training filters ──────────────────────────────────────────────
    filtered = df.copy()

    # Date range filter — replaces train_window
    filtered = filtered[
        (filtered.index.date >= date_range[0]) &
        (filtered.index.date <= date_range[1])
    ]

    # Hour filter
    filtered = filtered[
        (filtered.index.hour >= hour_range[0]) &
        (filtered.index.hour <= hour_range[1])
    ]

    # Day type filter
    if day_type == "Weekday":
        if "weekend" in filtered.columns and "holiday" in filtered.columns:
            filtered = filtered[
                (filtered["weekend"] == 0) & (filtered["holiday"] == 0)
            ]
    elif day_type == "Weekend":
        if "weekend" in filtered.columns:
            filtered = filtered[filtered["weekend"] == 1]
    elif day_type == "Holiday":
        if "holiday" in filtered.columns:
            filtered = filtered[filtered["holiday"] == 1]

    # Season filter
    SEASON_MONTHS = {
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
        "Winter": [12, 1, 2],
    }
    if season != "All":
        filtered = filtered[
            filtered.index.month.isin(SEASON_MONTHS[season])
        ]

    st.divider()

    # ── filtered data summary ──────────────────────────────────────────────
    st.caption(
        f"Training dataset: **{len(filtered):,} hours** · "
        f"Region: **{region.capitalize()}** · "
        f"Date range: **{date_range[0].strftime('%b %Y')} → {date_range[1].strftime('%b %Y')}** · "
        f"Hours: **{hour_range[0]}:00 – {hour_range[1]}:00** · "
        f"Day type: **{day_type}** · "
        f"Season: **{season}**"
    )

    # ── train button ───────────────────────────────────────────────────────
    with st.container(border=True):
        train_clicked = st.button(
            "Train / Refresh Model",
            type="primary",
            use_container_width=True,
            help="Trains XGBoost on the filtered dataset. Takes 5–15 seconds.",
        )

    if train_clicked:
        with st.spinner("Training XGBoost model — this takes a few seconds..."):
            try:
                results = train_model(filtered, target_col)
                st.session_state["driver_results"] = results
                st.session_state["driver_results"]["date_range"] = date_range
                st.session_state["driver_results"]["hour_range"] = hour_range
                st.session_state["driver_results"]["day_type"] = day_type
                st.session_state["driver_results"]["season"] = season
                st.success(
                    f"Model trained on {results['n_train']:,} hours · "
                    f"tested on {results['n_test']:,} hours · "
                    f"{len(results['feature_names'])} features used"
                )
            except ValueError as e:
                st.error(f"Training failed: {e}")
                return
            except Exception as e:
                st.error(f"Unexpected error during training: {e}")
                return

    # ── display results ────────────────────────────────────────────────────
    if "driver_results" not in st.session_state:
        st.info(
            "Configure the model above and click **Train / Refresh Model** "
            "to begin. Training typically takes 5–15 seconds."
        )
        return

    results = st.session_state["driver_results"]

    # ── stale model warning ────────────────────────────────────────────────
    if (results["target_col"] != target_col
        or results.get("date_range") != date_range
        or results.get("hour_range") != hour_range
        or results.get("day_type") != day_type
        or results.get("season") != season):
        st.warning(
            "Model configuration has changed since last training. "
            "Click **Train / Refresh Model** to update results."
        )

    # ── metrics strip ──────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="font-family: \'Courier New\', monospace; font-size: 1.1rem; '
        'color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; '
        'margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; '
        'padding-bottom: 0.5rem;">Model Performance</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE",            f"${results['mae']:.2f}/MWh")
        m2.metric("RMSE",           f"${results['rmse']:.2f}/MWh")
        m3.metric(
            "Direction Acc",
            f"{results['direction_acc']:.1f}%"
            if results["direction_acc"] is not None
            else "N/A",
        )
        m4.metric("Train Hours",    f"{results['n_train']:,}")
        m5.metric("Test Hours",     f"{results['n_test']:,}")

    st.divider()

    # ── auto summary ───────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family: \'Courier New\', monospace; font-size: 1.1rem; '
        'color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; '
        'margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; '
        'padding-bottom: 0.5rem;">Auto Summary</div>',
        unsafe_allow_html=True,
    )
    summary = generate_summary(
        results["importance_df"],
        results["target_col"],
        results["mae"],
        results["rmse"],
        results["direction_acc"],
    )

    with st.container(border=True):
        st.markdown(summary)

    st.divider()

    # ── feature importance chart ───────────────────────────────────────────
    st.markdown(
        '<div style="font-family: \'Courier New\', monospace; font-size: 1.1rem; '
        'color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; '
        'margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; '
        'padding-bottom: 0.5rem;">Top N Feature Importances</div>',
        unsafe_allow_html=True,
    )

    imp_df = results["importance_df"].head(top_n_features).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(
            color=imp_df["Importance"],
            colorscale="Blues",
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Feature Importance — {results['target_col']}",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#1e2d40", title="Importance (Gain)"),
        yaxis=dict(gridcolor="#1e2d40", tickfont=dict(size=11)),
        margin=dict(l=200, r=40, t=60, b=40),
        height=max(350, top_n_features * 38),
    )

    with st.container(border=True):
        st.markdown(
            '<div style="font-family: \'Courier New\', monospace; font-size: 0.85rem; '
            'color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; '
            'margin-bottom: 0.75rem;">Feature Importance</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── full importance table ──────────────────────────────────────────────
    with st.expander("View Full Feature Importance Table"):
        st.dataframe(
            results["importance_df"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Importance": st.column_config.ProgressColumn(
                    "Importance",
                    format="%.4f",
                    min_value=0,
                    max_value=float(
                        results["importance_df"]["Importance"].max()
                    ),
                )
            }
        )
        st.download_button(
            label="⬇️ Export Feature Importance CSV",
            data=results["importance_df"].to_csv(index=False).encode("utf-8"),
            file_name=f"feature_importance_{results['target_col']}.csv",
            mime="text/csv"
        )


main()
