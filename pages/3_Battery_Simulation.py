import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import load_region_df
from src.soc_engine import BatterySpec
from src.forecaster import (
    BATTERY_LZ_COL,
    BATTERY_REGION,
    train_houston_price_model,
    generate_forecast_range,
    forecast_accuracy,
)
from src.dam_bidder import (
    run_dam_campaign,
    dam_campaign_kpis,
)
from src.forecast_error import (
    run_perfect_foresight_campaign,
    compute_forecast_error_analysis,
    forecast_error_summary,
    monthly_error_breakdown,
    capture_ratio_distribution,
)
from src.risk import (
    compute_risk_metrics,
    compute_seasonal_reliability,
    compute_drawdown_series,
)
from src.charts import (
    cumulative_revenue_with_drawdown,
    monthly_revenue_box_plot,
)


def _section_header(label: str) -> None:
    st.markdown(
        f"<div style=\"font-family: 'Courier New', monospace; font-size: 1.1rem; "
        f"color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; "
        f"margin: 2rem 0 1rem 0; border-bottom: 1px solid #30363d; "
        f"padding-bottom: 0.5rem;\">{label}</div>",
        unsafe_allow_html=True,
    )


def _card_label(label: str) -> None:
    st.markdown(
        f"<div style=\"font-family: 'Courier New', monospace; font-size: 0.85rem; "
        f"color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em; "
        f"margin-bottom: 0.75rem;\">{label}</div>",
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
                    Battery Simulation
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
        "Houston-located BESS operating in the ERCOT Day-Ahead Market. "
        "Forecast-driven dispatch using XGBoost price forecasts submitted "
        "by 10:00 AM CPT the prior day. All revenue settled at LZ_HOUSTON_DAM. "
        "See the Assumptions page for full model boundaries."
    )

    # Load data
    df = load_region_df(BATTERY_REGION)

    # Section 1: Battery Parameters
    st.divider()
    _section_header("Battery Parameters")

    with st.container(border=True):
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            power_mw = st.slider(
                "Power Rating (MW)",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Rated charge and discharge power in MW",
            )
            duration_hours = st.selectbox(
                "Duration (hours)",
                options=[1, 2, 4, 6, 8],
                index=2,
                help="Storage duration — energy capacity = MW × hours",
            )

        with p_col2:
            charge_eff = (
                st.slider(
                    "Charge Efficiency (%)",
                    min_value=80,
                    max_value=98,
                    value=92,
                    step=1,
                    help="One-way charge efficiency — losses occur on the way in",
                )
                / 100
            )

            discharge_eff = (
                st.slider(
                    "Discharge Efficiency (%)",
                    min_value=80,
                    max_value=98,
                    value=92,
                    step=1,
                    help=(
                        "One-way discharge efficiency — losses occur on the way out"
                    ),
                )
                / 100
            )

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Energy Capacity", f"{power_mw * duration_hours:,} MWh")
        m2.metric(
            "Round-Trip Efficiency", f"{charge_eff * discharge_eff * 100:.1f}%"
        )
        m3.metric("Min SoC Buffer", "10%")
        m4.metric("Max Cycles / Day", "1")

    spec = BatterySpec(
        power_mw=power_mw,
        duration_hours=duration_hours,
        charge_eff=charge_eff,
        discharge_eff=discharge_eff,
        min_soc_pct=10.0,
        max_soc_pct=100.0,
        initial_soc_pct=50.0,
    )

    # Section 2: Model Configuration
    st.divider()
    _section_header("Model Configuration")
    st.caption(
        "The XGBoost model is trained on historical data up to the "
        "**Training Cutoff** date. The battery then operates in the "
        "**Simulation Window** using only information available at "
        "10:00 AM CPT each day — no lookahead."
    )

    with st.container(border=True):
        cfg_left, cfg_right = st.columns(2)

        with cfg_left:
            min_date = df.index.min().date()
            max_date = df.index.max().date()

            train_cutoff = st.date_input(
                "Training Cutoff Date",
                value=min(pd.Timestamp("2025-01-01").date(), max_date),
                min_value=min_date,
                max_value=max_date,
                help=(
                    "Model trained on all data BEFORE this date. "
                    "Simulation runs on dates ON OR AFTER this date. "
                    "Earlier cutoff = more training data but older model."
                ),
            )

        with cfg_right:
            default_start = train_cutoff if train_cutoff <= max_date else min_date
            default_end = max_date
            sim_start_in = st.date_input(
                "Simulation Start",
                value=default_start,
                min_value=min_date,
                max_value=max_date,
                help="First operating day of the simulation.",
            )
            sim_end_in = st.date_input(
                "Simulation End",
                value=default_end,
                min_value=sim_start_in,
                max_value=max_date,
                help="Last operating day of the simulation.",
            )

    # Validate simulation window starts on or after training cutoff
    if sim_start_in < train_cutoff:
        st.error(
            "Simulation start must be on or after the training cutoff. "
            "The model cannot operate on data it was trained on."
        )
        st.stop()

    sim_start = pd.Timestamp(sim_start_in)
    sim_end = pd.Timestamp(sim_end_in)
    test_start = pd.Timestamp(train_cutoff)

    # Training dataset summary
    train_hours = len(df[df.index < test_start])
    sim_days = (sim_end - sim_start).days + 1

    st.caption(
        f"Training data: **{train_hours:,} hours** "
        f"({min_date.strftime('%b %Y')} → {train_cutoff.strftime('%b %Y')})  ·  "
        f"Simulation: **{sim_days} days** "
        f"({sim_start.strftime('%b %Y')} → {sim_end.strftime('%b %Y')})"
    )

    # Section 3: Run Simulation
    st.divider()
    _section_header("Run Simulation")

    with st.container(border=True):
        run_button = st.button(
            "Train Model & Run Simulation",
            type="primary",
            use_container_width=True,
        )

        sim_params = {
            "power_mw": power_mw,
            "duration_hours": duration_hours,
            "charge_eff": charge_eff,
            "discharge_eff": discharge_eff,
            "train_cutoff": str(train_cutoff),
            "sim_start": str(sim_start),
            "sim_end": str(sim_end),
        }

        if (
            "sim_results" in st.session_state
            and st.session_state.get("sim_params") != sim_params
        ):
            st.warning(
                "Configuration has changed since the last run. "
                "Click **Train Model & Run Simulation** to refresh results."
            )

        if run_button:
            with st.spinner(
                "Step 1/4 — Training XGBoost price forecast model..."
            ):
                model_result = train_houston_price_model(df, test_start)

            with st.spinner(
                "Step 2/4 — Generating day-ahead forecasts..."
            ):
                forecast_df = generate_forecast_range(
                    model_result, df, sim_start, sim_end
                )

            with st.spinner(
                "Step 3/4 — Running DAM bidding campaign..."
            ):
                dam_campaign = run_dam_campaign(forecast_df, df, spec)

            with st.spinner(
                "Step 4/4 — Running forecast error analysis..."
            ):
                pf_campaign = run_perfect_foresight_campaign(
                    df[df.index >= sim_start].copy(), spec
                )
                analysis_df = compute_forecast_error_analysis(
                    pf_campaign, dam_campaign
                )

            st.session_state["sim_results"] = {
                "model_result": model_result,
                "forecast_df": forecast_df,
                "dam_campaign": dam_campaign,
                "pf_campaign": pf_campaign,
                "analysis_df": analysis_df,
            }
            st.session_state["sim_params"] = sim_params
            st.info(
                f"Simulation complete — "
                f"{len(dam_campaign)} operating days and "
                f"{len(forecast_df):,} forecast hours."
            )

    # Section 4: Results
    if "sim_results" not in st.session_state:
        st.info(
            "Configure battery parameters and model settings above, "
            "then click **Train Model & Run Simulation** to begin."
        )
        return

    results = st.session_state["sim_results"]
    model_result = results["model_result"]
    forecast_df = results["forecast_df"]
    dam_campaign = results["dam_campaign"]
    pf_campaign = results["pf_campaign"]
    analysis_df = results["analysis_df"]

    st.divider()
    _section_header("Results")

    # 4a: Forecast Model Quality
    _section_header("Forecast Model Quality")
    st.caption(
        "XGBoost model trained to predict LZ_HOUSTON_DAM hourly prices. "
        "Features lagged 24 hours — no lookahead."
    )

    acc = forecast_accuracy(forecast_df)
    kpis = dam_campaign_kpis(dam_campaign, spec)

    with st.container(border=True):
        fq1, fq2, fq3 = st.columns(3)
        fq1.metric(
            "Forecast MAE",
            f"${acc['mae']:.2f}/MWh",
            help="Mean absolute error of hourly LZ_HOUSTON_DAM forecast vs actual",
        )
        fq2.metric(
            "Forecast RMSE",
            f"${acc['rmse']:.2f}/MWh",
            help="Root mean squared error — penalises large misses more heavily",
        )
        fq3.metric(
            "Hours Forecasted",
            f"{acc['n_hours']:,}",
            help="Total hours in simulation window with forecast and actual prices",
        )

    # Forecast vs actual scatter
    with st.expander("Forecast vs Actual Price Chart"):
        with st.container(border=True):
            _card_label("Forecast vs Actual")
            sample = forecast_df.dropna()
            if len(sample) > 2000:
                sample = sample.sample(2000, random_state=42)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Scatter(
                    x=sample["actual_houston_price"],
                    y=sample["forecast_houston_price"],
                    mode="markers",
                    marker=dict(size=3, opacity=0.4, color="#38bdf8"),
                    name="Forecast vs Actual",
                )
            )
            price_range = [
                sample["actual_houston_price"].min(),
                sample["actual_houston_price"].max(),
            ]
            fig_scatter.add_trace(
                go.Scatter(
                    x=price_range,
                    y=price_range,
                    mode="lines",
                    line=dict(color="#94a3b8", dash="dash", width=1),
                    name="Perfect Forecast",
                )
            )
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e17",
                plot_bgcolor="#111827",
                height=400,
                xaxis_title="Actual LZ_HOUSTON_DAM ($/MWh)",
                yaxis_title="Forecast LZ_HOUSTON_DAM ($/MWh)",
                title="Forecast vs Actual — Random Sample of Hours",
                font=dict(family="monospace", color="#94a3b8", size=11),
                xaxis=dict(gridcolor="#1e2d40"),
                yaxis=dict(gridcolor="#1e2d40"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # 4b: Forecast Error Analysis — The Three Numbers
    st.divider()
    _section_header("Forecast Error Analysis")
    st.caption(
        "The three numbers that matter: what the battery could have earned "
        "with perfect information, what it actually earned using the forecast, "
        "and the dollar cost of imperfect forecasting."
    )

    summary = forecast_error_summary(analysis_df, spec)
    dist = capture_ratio_distribution(analysis_df)

    st.markdown("#### The Three Numbers")
    with st.container(border=True):
        n1, n2, n3, n4 = st.columns(4)

        n1.metric(
            "Perfect Foresight Revenue",
            f"${summary['total_pf_revenue']:,.0f}",
            help=(
                "Upper bound — unknowable in practice. "
                "Battery charges at exact cheapest hours and discharges "
                "at exact most expensive hours each day using actual prices."
            ),
        )
        n2.metric(
            "Forecast-Driven Revenue",
            f"${summary['total_dam_revenue']:,.0f}",
            delta=f"${summary['total_dam_revenue'] - summary['total_pf_revenue']:,.0f}",
            help=(
                "Realistic estimate. Battery dispatches based on XGBoost "
                "price forecasts submitted by 10AM the prior day. "
                "Settled at actual LZ_HOUSTON_DAM prices."
            ),
        )
        n3.metric(
            "Forecast Error Cost",
            f"${summary['total_forecast_error_cost']:,.0f}",
            help=(
                "Direct dollar cost of imperfect forecasting. "
                "= Perfect Foresight Revenue - Forecast-Driven Revenue. "
                "This is what better forecasting infrastructure is worth."
            ),
        )
        n4.metric(
            "Capture Ratio",
            f"{summary['overall_capture_ratio']:.1f}%",
            help=(
                "Forecast-Driven / Perfect Foresight × 100. "
                "100% = perfect forecast. 0% = forecast adds no value. "
                "Negative = forecast caused net losses."
            ),
        )

    # Per MW / Year metrics
    st.markdown("#### Per MW / Year (Size-Adjusted)")
    with st.container(border=True):
        mw1, mw2, mw3 = st.columns(3)
        mw1.metric(
            "PF Revenue / MW / Year",
            f"${summary['pf_revenue_per_mw_year']:,.0f}",
            help="Comparable across battery sizes",
        )
        mw2.metric(
            "DAM Revenue / MW / Year",
            f"${summary['dam_revenue_per_mw_year']:,.0f}",
        )
        mw3.metric(
            "Error Cost / MW / Year",
            f"${summary['error_cost_per_mw_year']:,.0f}",
            help="Annual value of improving the forecast by 1 percentage point",
        )

    # Daily Capture Ratio Distribution
    st.markdown("#### Daily Capture Ratio Distribution")
    with st.container(border=True):
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Excellent >90%", f"{dist['pct_days_excellent']:.1f}%")
        d2.metric("Good 70–90%", f"{dist['pct_days_good']:.1f}%")
        d3.metric("Moderate 50–70%", f"{dist['pct_days_moderate']:.1f}%")
        d4.metric("Poor 0–50%", f"{dist['pct_days_poor']:.1f}%")
        d5.metric(
            "Negative <0%",
            f"{dist['pct_days_negative']:.1f}%",
            help="Days where forecast caused net losses",
        )

    # ── Risk Analysis Section ────────────────────────────────────────────────
    st.divider()
    _section_header("Risk Analysis")
    st.caption(
        "Downside risk metrics from the forecast-driven DAM campaign. "
        "Helps assess revenue reliability and worst-case scenarios."
    )

    risk = compute_risk_metrics(analysis_df)

    with st.container(border=True):
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric(
            "Daily VaR (5%)",
            f"${risk['daily_var_5pct']:,.0f}",
            help="5th percentile daily revenue — on 95% of days you earn more than this.",
        )
        r2.metric(
            "CVaR / Exp. Shortfall",
            f"${risk['cvar_5pct']:,.0f}",
            help="Average revenue on the worst 5% of days.",
        )
        r3.metric(
            "Max Drawdown",
            f"${risk['max_drawdown']:,.0f}",
            help=(
                "Largest peak-to-trough decline in cumulative revenue "
                f"({risk['max_drawdown_days']} days)."
            ),
        )
        r4.metric(
            "Sharpe Ratio",
            f"{risk['sharpe_ratio']:.2f}",
            help=(
                "Annualised risk-adjusted return (mean / std × sqrt(252)). "
                "Higher is better."
            ),
        )
        r5.metric(
            "Longest Losing Streak",
            f"{risk['longest_losing_streak']} days",
            help="Maximum consecutive days with zero or negative revenue.",
        )

    dd_series = compute_drawdown_series(analysis_df)
    with st.container(border=True):
        _card_label("Cumulative Revenue & Drawdown")
        fig_dd = cumulative_revenue_with_drawdown(dd_series)
        st.plotly_chart(fig_dd, use_container_width=True)

    with st.container(border=True):
        _card_label("Seasonal Reliability — Daily Revenue by Month")
        fig_box = monthly_revenue_box_plot(analysis_df)
        st.plotly_chart(fig_box, use_container_width=True)

    seasonal = compute_seasonal_reliability(analysis_df)
    with st.expander("Seasonal Reliability Table"):
        with st.container(border=True):
            st.dataframe(
                seasonal,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Month": st.column_config.TextColumn("Month", width="small"),
                    "Mean ($)": st.column_config.NumberColumn(format="$%,.0f"),
                    "Std ($)": st.column_config.NumberColumn(format="$%,.0f"),
                    "Min ($)": st.column_config.NumberColumn(format="$%,.0f"),
                    "Max ($)": st.column_config.NumberColumn(format="$%,.0f"),
                    "Days": st.column_config.NumberColumn(format="%d"),
                    "Reliable": st.column_config.CheckboxColumn(
                        "Reliable",
                        help="Month where std < mean and mean > 0 (consistent revenue).",
                    ),
                },
            )

    # Revenue Comparison Chart
    _section_header("DAM Campaign Details")

    with st.container(border=True):
        _card_label("Daily Revenue — Perfect Foresight vs Forecast-Driven")
        fig_rev = make_subplots(specs=[[{"secondary_y": True}]])

        fig_rev.add_trace(
            go.Bar(
                x=analysis_df["date"],
                y=analysis_df["pf_net_revenue"],
                name="Perfect Foresight",
                marker_color="#94a3b8",
                opacity=0.5,
            ),
            secondary_y=False,
        )

        fig_rev.add_trace(
            go.Bar(
                x=analysis_df["date"],
                y=analysis_df["dam_revenue"],
                name="Forecast-Driven DAM",
                marker_color="#38bdf8",
                opacity=0.8,
            ),
            secondary_y=False,
        )

        fig_rev.add_trace(
            go.Scatter(
                x=analysis_df["date"],
                y=analysis_df["capture_ratio"],
                name="Capture Ratio (%)",
                mode="lines",
                line=dict(color="#58a6ff", width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )

        fig_rev.add_hline(y=0, line_color="#475569", line_width=1)
        fig_rev.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#111827",
            height=420,
            barmode="overlay",
            legend=dict(orientation="h", y=1.08),
            xaxis_title="Date",
            font=dict(family="monospace", color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#1e2d40"),
            yaxis=dict(gridcolor="#1e2d40"),
        )
        fig_rev.update_yaxes(title_text="Daily Revenue ($)", secondary_y=False)
        fig_rev.update_yaxes(title_text="Capture Ratio (%)", secondary_y=True)

        st.plotly_chart(fig_rev, use_container_width=True)

    # Forecast Error Cost Chart
    with st.container(border=True):
        _card_label("Daily Forecast Error Cost")
        fig_err = go.Figure()

        err_colors = ["#58a6ff"] * len(analysis_df)

        fig_err.add_trace(
            go.Bar(
                x=analysis_df["date"],
                y=analysis_df["forecast_error_cost"],
                name="Forecast Error Cost ($)",
                marker_color=err_colors,
                opacity=0.8,
            )
        )

        fig_err.add_hline(y=0, line_color="#475569", line_width=1)
        fig_err.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#111827",
            height=320,
            xaxis_title="Date",
            yaxis_title="Forecast Error Cost ($)",
            showlegend=False,
            font=dict(family="monospace", color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#1e2d40"),
            yaxis=dict(gridcolor="#1e2d40"),
        )
        fig_err.add_annotation(
            text=(
                f"Total error cost: ${summary['total_forecast_error_cost']:,.0f}  ·  "
                f"Avg daily: ${summary['avg_daily_error_cost']:,.2f}  ·  "
                f"Worst day: ${summary['worst_error_cost']:,.2f} "
                f"({summary['worst_error_cost_day']})"
            ),
            xref="paper",
            yref="paper",
            x=0,
            y=1.12,
            showarrow=False,
            font=dict(size=11, color="#94a3b8"),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # DAM Campaign KPIs
    dam_kpis = dam_campaign_kpis(dam_campaign, spec)
    with st.container(border=True):
        k1, k2, k3 = st.columns(3)
        k1.metric("Profitable Days", f"{dam_kpis['profitable_days_pct']:.1f}%")
        k2.metric("Best Day", f"${dam_kpis['best_day']:,.2f}")
        k3.metric("Worst Day", f"${dam_kpis['worst_day']:,.2f}")

    # Monthly Breakdown
    monthly = monthly_error_breakdown(analysis_df)
    with st.container(border=True):
        _card_label("Monthly Breakdown")
        st.dataframe(
            monthly,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Month": st.column_config.TextColumn("Month", width="small"),
                "PF Revenue ($)": st.column_config.NumberColumn(format="$%,.0f"),
                "DAM Revenue ($)": st.column_config.NumberColumn(format="$%,.0f"),
                "Forecast Error Cost ($)": st.column_config.NumberColumn(
                    format="$%,.0f"
                ),
                "Avg Capture Ratio (%)": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100
                ),
                "Days": st.column_config.NumberColumn(format="%d"),
            },
        )

    # Full Daily Table
    with st.expander("Full Daily Results Table"):
        with st.container(border=True):
            _card_label("Daily Results")
            daily_display = analysis_df.copy()
            daily_display["date"] = daily_display["date"].astype(str)

            st.dataframe(
                daily_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "Date",
                    "pf_net_revenue": st.column_config.NumberColumn(
                        "PF Revenue ($)", format="$%,.2f"
                    ),
                    "dam_revenue": st.column_config.NumberColumn(
                        "DAM Revenue ($)", format="$%,.2f"
                    ),
                    "forecast_error_cost": st.column_config.NumberColumn(
                        "Error Cost ($)", format="$%,.2f"
                    ),
                    "capture_ratio": st.column_config.ProgressColumn(
                        "Capture Ratio (%)",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "idle_day": st.column_config.CheckboxColumn(
                        "Idle", help="Battery did not dispatch"
                    ),
                },
            )

            csv = daily_display.to_csv(index=False)
            st.download_button(
                "Export Daily Results CSV",
                data=csv,
                file_name="battery_sim_daily_results.csv",
                mime="text/csv",
            )

    # Methodology Note
    with st.expander("Methodology"):
        st.markdown(
            f"""
    **Strategy: Forecast-Driven DAM Self-Schedule**

    1. **Model Training**: XGBoost trained on historical data before
       `{train_cutoff}`. Target: `LZ_HOUSTON_DAM` hourly price.
       Features lagged 24 hours to eliminate lookahead bias.

    2. **Day-Ahead Forecast**: For each operating day D, the model
       generates 24 hourly price forecasts using Day D-1 features —
       simulating a 10:00 AM CPT bid submission deadline.

    3. **Bid Optimisation**: The battery selects one charge window and
       one discharge window of `{duration_hours}h` each that maximise
       expected revenue under the forecast. Discharge must start after
       charge ends (causal constraint). Idle bid submitted if no
       profitable combination found.

    4. **Settlement**: The bid is locked in at DAM prices. Revenue settled
       exclusively at actual `LZ_HOUSTON_DAM` prices regardless of
       forecast accuracy.

    5. **Perfect Foresight Benchmark**: For each day, the optimal
       charge/discharge windows are computed using actual prices —
       representing the maximum achievable revenue with perfect information.

    6. **Forecast Error Cost**: The difference between perfect foresight
       and forecast-driven revenue. This is the dollar value of forecast
       uncertainty — and by implication the value of improving the model.

    **Key Assumptions**
    - Battery located in Houston — settles at `LZ_HOUSTON_DAM` only
    - Price taker — no market power
    - One cycle per day — one charge window, one discharge window
    - No ancillary services — energy arbitrage only
    - No O&M or degradation costs
    - See Assumptions page for full boundary conditions
    """
        )


if __name__ == "__main__":
    main()

