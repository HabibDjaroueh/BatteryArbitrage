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

# ── Page Config ────────────────────────────────────────────────────────────
st.title("Battery Simulation")
st.caption(
    "Houston-located BESS operating in the ERCOT Day-Ahead Market. "
    "Forecast-driven dispatch using XGBoost price forecasts submitted "
    "by 10:00 AM CPT the prior day. All revenue settled at LZ_HOUSTON_DAM. "
    "See the Assumptions page for full model boundaries."
)

# ── Sidebar — Region locked to Houston ────────────────────────────────────
st.sidebar.title("Battery Configuration")
st.sidebar.info(
    "🔋 Battery location is fixed to **Houston (LZ_HOUSTON_DAM)**. "
    "The forecast model predicts Houston DAM prices. "
    "All revenue is settled at the Houston DAM zonal price."
)

# ── Load Data ──────────────────────────────────────────────────────────────
df = load_region_df(BATTERY_REGION)

# ── Section 1: Battery Parameters ─────────────────────────────────────────
st.divider()
st.subheader("🔋 Battery Parameters")

p_col1, p_col2 = st.columns(2)

with p_col1:
    power_mw = st.slider(
        "Power Rating (MW)",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Rated charge and discharge power in MW"
    )
    duration_hours = st.selectbox(
        "Duration (hours)",
        options=[1, 2, 4, 6, 8],
        index=2,
        help="Storage duration — energy capacity = MW × hours"
    )

with p_col2:
    charge_eff = st.slider(
        "Charge Efficiency (%)",
        min_value=80,
        max_value=98,
        value=92,
        step=1,
        help="One-way charge efficiency — losses occur on the way in"
    ) / 100

    discharge_eff = st.slider(
        "Discharge Efficiency (%)",
        min_value=80,
        max_value=98,
        value=92,
        step=1,
        help="One-way discharge efficiency — losses occur on the way out"
    ) / 100

# Live derived metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Energy Capacity",    f"{power_mw * duration_hours:,} MWh")
m2.metric("Round-Trip Efficiency", f"{charge_eff * discharge_eff * 100:.1f}%")
m3.metric("Min SoC Buffer",     "10%")
m4.metric("Max Cycles / Day",   "1")

spec = BatterySpec(
    power_mw=power_mw,
    duration_hours=duration_hours,
    charge_eff=charge_eff,
    discharge_eff=discharge_eff,
    min_soc_pct=10.0,
    max_soc_pct=100.0,
    initial_soc_pct=50.0,
)

# ── Section 2: Model Configuration ────────────────────────────────────────
st.divider()
st.subheader("⚙️ Model Configuration")
st.markdown(
    "The XGBoost model is trained on historical data up to the "
    "**Training Cutoff** date. The battery then operates in the "
    "**Simulation Window** using only information available at "
    "10:00 AM CPT each day — no lookahead."
)

cfg_left, cfg_right = st.columns(2)

with cfg_left:
    min_date  = df.index.min().date()
    max_date  = df.index.max().date()

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
    default_end   = max_date
    sim_start_in  = st.date_input(
        "Simulation Start",
        value=default_start,
        min_value=min_date,
        max_value=max_date,
        help="First operating day of the simulation.",
    )
    sim_end_in    = st.date_input(
        "Simulation End",
        value=default_end,
        min_value=sim_start_in,
        max_value=max_date,
        help="Last operating day of the simulation.",
    )

# Validate simulation window starts on or after training cutoff
if sim_start_in < train_cutoff:
    st.error(
        "⚠️ Simulation start must be on or after the training cutoff. "
        "The model cannot operate on data it was trained on."
    )
    st.stop()

sim_start = pd.Timestamp(sim_start_in)
sim_end   = pd.Timestamp(sim_end_in)
test_start = pd.Timestamp(train_cutoff)

# Training dataset summary
train_hours = len(df[df.index < test_start])
sim_days    = (sim_end - sim_start).days + 1

st.caption(
    f"Training data: **{train_hours:,} hours** "
    f"({min_date.strftime('%b %Y')} → {train_cutoff.strftime('%b %Y')})  ·  "
    f"Simulation: **{sim_days} days** "
    f"({sim_start.strftime('%b %Y')} → {sim_end.strftime('%b %Y')})"
)

# ── Section 3: Run Simulation ──────────────────────────────────────────────
# Single button runs all four steps in order. Each step runs inside a spinner
# so the user sees progress. Results are stored in session state so they
# persist across reruns (e.g. expanding an expander) without re-running.
# A "stale" warning is shown if any parameter changes after a run, prompting
# the user to click the button again to refresh results.
st.divider()

run_col, _ = st.columns([1, 2])
with run_col:
    run_button = st.button(
        "🚀 Train Model & Run Simulation",
        type="primary",
        use_container_width=True,
    )

# Snapshot of current UI parameters. Compared to session state to detect
# whether the user changed battery config or model config since last run.
sim_params = {
    "power_mw":       power_mw,
    "duration_hours": duration_hours,
    "charge_eff":     charge_eff,
    "discharge_eff":  discharge_eff,
    "train_cutoff":   str(train_cutoff),
    "sim_start":      str(sim_start),
    "sim_end":        str(sim_end),
}

# Stale warning: show only when we have prior results and current params
# differ from the params that were used for that run.
if (
    "sim_results" in st.session_state
    and st.session_state.get("sim_params") != sim_params
):
    st.warning(
        "⚠️ Configuration has changed since last run. "
        "Click **🚀 Train Model & Run Simulation** to update."
    )

if run_button:
    # Step 1/4 — Train XGBoost on data before train_cutoff; target LZ_HOUSTON_DAM.
    with st.spinner("Step 1/4 — Training XGBoost price forecast model..."):
        model_result = train_houston_price_model(df, test_start)

    # Step 2/4 — Hourly forecasts for sim window using D-1 features (no lookahead).
    with st.spinner("Step 2/4 — Generating day-ahead forecasts..."):
        forecast_df = generate_forecast_range(
            model_result, df, sim_start, sim_end
        )

    # Step 3/4 — For each day: optimise bid from forecast, settle at actual LZ_HOUSTON_DAM.
    with st.spinner("Step 3/4 — Running DAM bidding campaign..."):
        dam_campaign = run_dam_campaign(forecast_df, df, spec)

    # Step 4/4 — Perfect foresight benchmark + merge with DAM; forecast error cost & capture ratio.
    with st.spinner("Step 4/4 — Running forecast error analysis..."):
        pf_campaign  = run_perfect_foresight_campaign(
            df[df.index >= sim_start].copy(), spec
        )
        analysis_df  = compute_forecast_error_analysis(
            pf_campaign, dam_campaign
        )

    # Persist in session state so results survive reruns; keys used below in Section 4.
    st.session_state["sim_results"] = {
        "model_result": model_result,  # trained model, feature_names, predictions_df, mae, rmse, etc.
        "forecast_df":  forecast_df,  # hourly forecast_houston_price & actual_houston_price
        "dam_campaign": dam_campaign, # one row per day: net_revenue, forecast_revenue, realisation_gap, ...
        "pf_campaign":  pf_campaign,  # one row per day: pf_net_revenue, pf_charge_cost, ...
        "analysis_df":  analysis_df,  # merged: date, pf_net_revenue, dam_revenue, forecast_error_cost, capture_ratio, idle_day
    }
    st.session_state["sim_params"] = sim_params
    st.success(
        f"✅ Simulation complete — "
        f"{len(dam_campaign)} operating days · "
        f"{len(forecast_df):,} forecast hours"
    )

# ── Section 4: Results ─────────────────────────────────────────────────────
# All result blocks read from st.session_state["sim_results"] so they stay
# in sync with the last run; no page currently uses run_arbitrage, run_rule_based,
# or run_sensitivity from src.battery (see deprecation notes there).
if "sim_results" not in st.session_state:
    st.info(
        "Configure battery parameters and model settings above, "
        "then click **🚀 Train Model & Run Simulation** to begin."
    )
    st.stop()

results      = st.session_state["sim_results"]
model_result = results["model_result"]
forecast_df  = results["forecast_df"]
dam_campaign = results["dam_campaign"]
pf_campaign  = results["pf_campaign"]
analysis_df  = results["analysis_df"]

# ── 4a: Forecast Model Quality ────────────────────────────────────────────
st.divider()
st.subheader("📡 Forecast Model Quality")
st.caption(
    "XGBoost model trained to predict LZ_HOUSTON_DAM hourly prices. "
    "Features lagged 24 hours — no lookahead. "
    "Direction accuracy measures how often the model correctly predicts "
    "whether Houston price is above or below the daily average — "
    "the signal used for dispatch decisions."
)

acc  = forecast_accuracy(forecast_df)
kpis = dam_campaign_kpis(dam_campaign, spec)

fq1, fq2, fq3, fq4 = st.columns(4)
fq1.metric(
    "Forecast MAE",
    f"${acc['mae']:.2f}/MWh",
    help="Mean absolute error of hourly LZ_HOUSTON_DAM forecast vs actual"
)
fq2.metric(
    "Forecast RMSE",
    f"${acc['rmse']:.2f}/MWh",
    help="Root mean squared error — penalises large misses more heavily"
)
fq3.metric(
    "Direction Accuracy",
    f"{acc['direction_accuracy']:.1f}%",
    help=(
        "% of hours where price direction was correctly predicted. "
        "50% = random. 65%+ = useful signal. 75%+ = strong."
    )
)
fq4.metric(
    "Hours Forecasted",
    f"{acc['n_hours']:,}",
    help="Total hours in simulation window with forecast and actual prices"
)

# Forecast vs actual scatter
with st.expander("📊 Forecast vs Actual Price Chart"):
    sample = forecast_df.dropna()
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=42)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=sample["actual_houston_price"],
        y=sample["forecast_houston_price"],
        mode="markers",
        marker=dict(size=3, opacity=0.4, color="#38bdf8"),
        name="Forecast vs Actual",
    ))
    price_range = [
        sample["actual_houston_price"].min(),
        sample["actual_houston_price"].max()
    ]
    fig_scatter.add_trace(go.Scatter(
        x=price_range, y=price_range,
        mode="lines",
        line=dict(color="#f87171", dash="dash", width=1),
        name="Perfect Forecast",
    ))
    fig_scatter.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Actual LZ_HOUSTON_DAM ($/MWh)",
        yaxis_title="Forecast LZ_HOUSTON_DAM ($/MWh)",
        title="Forecast vs Actual — Random Sample of Hours",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── 4b: Forecast Error Analysis — The Three Numbers ───────────────────────
st.divider()
st.subheader("📊 Forecast Error Analysis")
st.caption(
    "The three numbers that matter: what the battery could have earned "
    "with perfect information, what it actually earned using the forecast, "
    "and the dollar cost of imperfect forecasting."
)

summary = forecast_error_summary(analysis_df, spec)
dist    = capture_ratio_distribution(analysis_df)

# ── The Three Numbers — hero metrics ──────────────────────────────────────
st.markdown("#### The Three Numbers")
n1, n2, n3, n4 = st.columns(4)

n1.metric(
    "Perfect Foresight Revenue",
    f"${summary['total_pf_revenue']:,.0f}",
    help=(
        "Upper bound — unknowable in practice. "
        "Battery charges at exact cheapest hours and discharges "
        "at exact most expensive hours each day using actual prices."
    )
)
n2.metric(
    "Forecast-Driven Revenue",
    f"${summary['total_dam_revenue']:,.0f}",
    delta=f"${summary['total_dam_revenue'] - summary['total_pf_revenue']:,.0f}",
    help=(
        "Realistic estimate. Battery dispatches based on XGBoost "
        "price forecasts submitted by 10AM the prior day. "
        "Settled at actual LZ_HOUSTON_DAM prices."
    )
)
n3.metric(
    "Forecast Error Cost",
    f"${summary['total_forecast_error_cost']:,.0f}",
    help=(
        "Direct dollar cost of imperfect forecasting. "
        "= Perfect Foresight Revenue - Forecast-Driven Revenue. "
        "This is what better forecasting infrastructure is worth."
    )
)
n4.metric(
    "Capture Ratio",
    f"{summary['overall_capture_ratio']:.1f}%",
    help=(
        "Forecast-Driven / Perfect Foresight × 100. "
        "100% = perfect forecast. 0% = forecast adds no value. "
        "Negative = forecast caused net losses."
    )
)

# ── Per MW / Year metrics ──────────────────────────────────────────────────
st.markdown("#### Per MW / Year (Size-Adjusted)")
mw1, mw2, mw3 = st.columns(3)
mw1.metric(
    "PF Revenue / MW / Year",
    f"${summary['pf_revenue_per_mw_year']:,.0f}",
    help="Comparable across battery sizes"
)
mw2.metric(
    "DAM Revenue / MW / Year",
    f"${summary['dam_revenue_per_mw_year']:,.0f}",
)
mw3.metric(
    "Error Cost / MW / Year",
    f"${summary['error_cost_per_mw_year']:,.0f}",
    help="Annual value of improving the forecast by 1 percentage point"
)

# ── Daily Capture Ratio Distribution ──────────────────────────────────────
st.markdown("#### Daily Capture Ratio Distribution")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Excellent >90%",  f"{dist['pct_days_excellent']:.1f}%")
d2.metric("Good 70–90%",     f"{dist['pct_days_good']:.1f}%")
d3.metric("Moderate 50–70%", f"{dist['pct_days_moderate']:.1f}%")
d4.metric("Poor 0–50%",     f"{dist['pct_days_poor']:.1f}%")
d5.metric("Negative <0%",   f"{dist['pct_days_negative']:.1f}%",
          help="Days where forecast caused net losses")

# ── Revenue Comparison Chart ───────────────────────────────────────────────
st.markdown("#### Daily Revenue — Perfect Foresight vs Forecast-Driven")

fig_rev = make_subplots(specs=[[{"secondary_y": True}]])

fig_rev.add_trace(go.Bar(
    x=analysis_df["date"],
    y=analysis_df["pf_net_revenue"],
    name="Perfect Foresight",
    marker_color="#94a3b8",
    opacity=0.5,
), secondary_y=False)

fig_rev.add_trace(go.Bar(
    x=analysis_df["date"],
    y=analysis_df["dam_revenue"],
    name="Forecast-Driven DAM",
    marker_color="#38bdf8",
    opacity=0.8,
), secondary_y=False)

fig_rev.add_trace(go.Scatter(
    x=analysis_df["date"],
    y=analysis_df["capture_ratio"],
    name="Capture Ratio (%)",
    mode="lines",
    line=dict(color="#fb923c", width=1.5),
), secondary_y=True)

fig_rev.add_hline(y=0, line_color="#475569", line_width=1)
fig_rev.update_layout(
    template="plotly_dark",
    height=420,
    barmode="overlay",
    legend=dict(orientation="h", y=1.08),
    xaxis_title="Date",
)
fig_rev.update_yaxes(title_text="Daily Revenue ($)",  secondary_y=False)
fig_rev.update_yaxes(title_text="Capture Ratio (%)",  secondary_y=True)

st.plotly_chart(fig_rev, use_container_width=True)

# ── Forecast Error Cost Chart ──────────────────────────────────────────────
st.markdown("#### Daily Forecast Error Cost")
fig_err = go.Figure()

err_colors = analysis_df["forecast_error_cost"].apply(
    lambda x: "#f87171" if x > 0 else "#4ade80"
).tolist()

fig_err.add_trace(go.Bar(
    x=analysis_df["date"],
    y=analysis_df["forecast_error_cost"],
    name="Forecast Error Cost ($)",
    marker_color=err_colors,
))

fig_err.add_hline(y=0, line_color="#475569", line_width=1)
fig_err.update_layout(
    template="plotly_dark",
    height=320,
    xaxis_title="Date",
    yaxis_title="Forecast Error Cost ($)",
    showlegend=False,
)
fig_err.add_annotation(
    text=(
        f"Total error cost: ${summary['total_forecast_error_cost']:,.0f}  ·  "
        f"Avg daily: ${summary['avg_daily_error_cost']:,.2f}  ·  "
        f"Worst day: ${summary['worst_error_cost']:,.2f} "
        f"({summary['worst_error_cost_day']})"
    ),
    xref="paper", yref="paper",
    x=0, y=1.12, showarrow=False,
    font=dict(size=11, color="#94a3b8"),
)
st.plotly_chart(fig_err, use_container_width=True)

# ── 4c: DAM Campaign Details ───────────────────────────────────────────────
st.divider()
st.subheader("📋 DAM Campaign Details")

dam_kpis = dam_campaign_kpis(dam_campaign, spec)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Profitable Days",
          f"{dam_kpis['profitable_days_pct']:.1f}%")
k2.metric("Best Day",
          f"${dam_kpis['best_day']:,.2f}")
k3.metric("Worst Day",
          f"${dam_kpis['worst_day']:,.2f}")
k4.metric("SoC Violations",
          f"{dam_kpis['total_soc_violations']}",
          help="Hours where physical SoC constraints clipped the bid schedule")

# ── Monthly Breakdown ──────────────────────────────────────────────────────
st.markdown("#### Monthly Breakdown")
monthly = monthly_error_breakdown(analysis_df)
st.dataframe(
    monthly,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Month":                   st.column_config.TextColumn("Month", width="small"),
        "PF Revenue ($)":          st.column_config.NumberColumn(format="$%,.0f"),
        "DAM Revenue ($)":         st.column_config.NumberColumn(format="$%,.0f"),
        "Forecast Error Cost ($)": st.column_config.NumberColumn(format="$%,.0f"),
        "Avg Capture Ratio (%)":   st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0, max_value=100
        ),
        "Days":                    st.column_config.NumberColumn(format="%d"),
    }
)

# ── Full Daily Table ───────────────────────────────────────────────────────
with st.expander("📄 Full Daily Results Table"):
    daily_display = analysis_df.copy()
    daily_display["date"] = daily_display["date"].astype(str)

    st.dataframe(
        daily_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date":                 "Date",
            "pf_net_revenue":       st.column_config.NumberColumn(
                "PF Revenue ($)", format="$%,.2f"),
            "dam_revenue":          st.column_config.NumberColumn(
                "DAM Revenue ($)", format="$%,.2f"),
            "forecast_error_cost":  st.column_config.NumberColumn(
                "Error Cost ($)", format="$%,.2f"),
            "capture_ratio":        st.column_config.ProgressColumn(
                "Capture Ratio (%)", format="%.1f%%",
                min_value=0, max_value=100),
            "idle_day":             st.column_config.CheckboxColumn(
                "Idle", help="Battery did not dispatch"),
        }
    )

    csv = daily_display.to_csv(index=False)
    st.download_button(
        "⬇️ Export Daily Results CSV",
        data=csv,
        file_name="battery_sim_daily_results.csv",
        mime="text/csv",
    )

# ── Methodology Note ───────────────────────────────────────────────────────
with st.expander("📖 Methodology"):
    st.markdown(f"""
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
    """)
