import streamlit as st

st.title("Model Assumptions")
st.caption(
    "This page documents every assumption underpinning the battery "
    "simulation. These boundaries define what the model is — and is not — "
    "claiming to represent. Transparency here is intentional: a model "
    "that understands its own limitations is more credible than one that doesn't."
)

# ── Section 1: Battery & Physical Setup ───────────────────────────────────
st.divider()
st.subheader("🔋 Battery & Physical Setup")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Location**
    The battery is assumed to be located in the Houston load zone
    (`LZ_HOUSTON_DAM`). All energy settlement occurs at the Houston
    DAM zonal price. Bus-level (nodal) pricing is not modelled —
    this introduces **basis risk** which is acknowledged but not quantified.

    **Technology**
    Lithium-ion battery energy storage system (BESS).
    Round-trip efficiency is user-configurable (default 85%).
    Efficiency is applied symmetrically — losses occur on both
    charge and discharge cycles.

    **Capacity**
    Power rating (MW) and duration (hours) are user-defined.
    Energy capacity = MW × hours. The battery cannot charge or
    discharge beyond its rated power in any single hour.
    """)

with col2:
    st.markdown("""
    **State of Charge (SoC)**
    SoC is bounded between 0% and 100% of energy capacity at all times.
    The battery begins each simulation period at 50% SoC.
    A minimum SoC buffer of 10% is maintained — the battery will not
    discharge below 10% SoC to protect battery health.

    **Degradation**
    Battery capacity degradation is **not modelled**.
    In practice, lithium-ion batteries lose approximately 2–3% of
    usable capacity per year under normal cycling conditions.
    Excluding degradation means long-run revenue estimates are
    slightly optimistic.

    **Interconnection**
    The battery is assumed to be fully interconnected with no
    transmission constraints or congestion costs.
    Grid connection costs and interconnection queue delays
    are outside the scope of this model.
    """)

# ── Section 2: Market Structure ────────────────────────────────────────────
st.divider()
st.subheader("🏛️ Market Structure")

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    **Market: ERCOT Day-Ahead Market (DAM)**
    The battery participates exclusively in the ERCOT Day-Ahead Market.
    Bids are assumed to be submitted by 10:00 AM CPT the day prior
    to the operating day, consistent with ERCOT DAM timelines.

    **Price Taker Assumption**
    The battery is modelled as a **strict price taker** — it has no
    market power and cannot influence clearing prices regardless of
    its size. This is a reasonable approximation for batteries under
    ~200 MW in ERCOT. For larger assets this assumption becomes
    increasingly aggressive, particularly during peak hours when
    battery dispatch represents a meaningful share of marginal supply.

    **Perfect Competition**
    The model assumes perfectly competitive markets with no strategic
    bidding, no counterparty risk, and no bid/offer spreads.
    All dispatch is assumed to clear at the DAM zonal price.
    """)

with col4:
    st.markdown("""
    **Settlement**
    Revenue is settled at `LZ_HOUSTON_DAM` zonal prices.
    Real-time market (RTM) participation and real-time settlement
    deviations are **not modelled**. In practice, batteries often
    participate in both DAM and RTM — excluding RTM understates
    both revenue opportunity and risk.

    **No Bilateral Contracts**
    All revenue is assumed to be merchant (spot market).
    Power Purchase Agreements (PPAs), tolling agreements, or
    other contracted revenue streams are not modelled.
    A contracted battery would have lower revenue variance but
    potentially lower upside than the merchant revenues shown here.

    **No Transmission Costs**
    Transmission and distribution charges, wheeling costs,
    and congestion rents are excluded from the revenue calculation.
    """)

# ── Section 3: Revenue Streams ─────────────────────────────────────────────
st.divider()
st.subheader("💰 Revenue Streams")

st.markdown("""
This model captures **energy arbitrage revenue only** — buying cheap,
selling expensive. This is one of several revenue streams available
to a battery in ERCOT. The table below shows what is and is not included:
""")

st.markdown("""
| Revenue Stream | Included | Notes |
|---|---|---|
| DAM Energy Arbitrage | ✅ Yes | Core model output |
| Real-Time Energy Arbitrage | ❌ No | RTM participation not modelled |
| Regulation Up (RegUp) | ❌ No | Significant revenue stream — excluded |
| Regulation Down (RegDown) | ❌ No | Excluded |
| Responsive Reserve Service (RRS) | ❌ No | Excluded |
| ERCOT Contingency Reserve Service (ECRS) | ❌ No | Introduced 2023 — excluded |
| Non-Spinning Reserve | ❌ No | Excluded |
| Capacity payments | ❌ No | ERCOT has no capacity market |
""")

st.info(
    "**Revenue Understatement Warning:** Ancillary services typically "
    "contribute 40–60% of total ERCOT battery revenue. A well-optimised "
    "battery co-optimising energy and ancillary services would earn "
    "approximately **1.4x–1.7x** the energy arbitrage revenue shown in "
    "this simulation. The figures presented here should be treated as a "
    "conservative lower bound on total achievable revenue."
)

# ── Section 4: Dispatch Strategy ──────────────────────────────────────────
st.divider()
st.subheader("⚡ Dispatch Strategy")

col5, col6 = st.columns(2)

with col5:
    st.markdown("""
    **Forecast-Driven DAM Strategy**
    The battery uses an XGBoost model trained on historical data
    to forecast next-day hourly prices for all four ERCOT zones.
    Dispatch decisions are made using these forecasts — not actual prices.
    The battery is committed to its DAM schedule and settles at
    DAM prices regardless of how actual prices evolve.

    Forecast error directly reduces revenue; the impact of
    uncertainty is reported explicitly in the simulation output.
    """)

with col6:
    st.markdown("""
    **Cycle Limit**
    The strategy is limited to one full charge/discharge cycle
    per day. Multi-cycle optimisation is not modelled.
    """)

# ── Section 5: Forecast Model ─────────────────────────────────────────────
st.divider()
st.subheader("🤖 Forecast Model")

col7, col8 = st.columns(2)

with col7:
    st.markdown("""
    **Algorithm**
    XGBoost gradient boosting regression. Trained on historical
    hourly ERCOT data with features including regional load,
    wind and solar generation, weather variables, heat index,
    net load, renewable penetration, and calendar features.

    **Training**
    Temporal train/test split — no shuffling. Test set is always
    the most recent 20% of the training window to prevent leakage.
    Training window, day type, season, and hour filters are
    user-configurable on the Drivers page.

    **Feature Lag**
    All features are lagged 24 hours relative to the prediction target.
    Features for operating day D are drawn from day D-1 actuals —
    eliminating lookahead bias. In a live system, day D weather forecasts
    and ERCOT load forecasts would replace the D-1 actuals, potentially
    improving accuracy. Using D-1 actuals is a conservative assumption
    that slightly understates what a production system would achieve.

    **Validation**
    Model performance is reported as MAE, RMSE, and direction
    accuracy (spread sign) on the held-out test set.
    Direction accuracy is the operationally relevant metric —
    it measures how often the model correctly predicts whether
    the spread will be positive or negative.
    """)

with col8:
    st.markdown("""
    **Limitations**
    - The model is trained on historical patterns and cannot
      anticipate structural market changes, new generator
      interconnections, or regulatory changes
    - Extreme price events (e.g. Winter Storm Uri) may not
      be well-represented in training data
    - The model predicts DAM zonal prices — it does not model
      real-time deviations or nodal price separation
    - Feature importance reflects correlation not causation —
      high wind generation importance reflects the structural
      relationship between renewables and price suppression
      in ERCOT but does not imply direct causality

    **Retraining**
    The model should be retrained periodically as market
    conditions evolve. Renewable penetration in ERCOT has
    increased significantly year-over-year — a model trained
    on 2022 data will underestimate the price impact of
    renewables in 2025.
    """)

# ── Section 6: Known Limitations ──────────────────────────────────────────
st.divider()
st.subheader("⚠️ Known Limitations & Caveats")

st.warning(
    "The following limitations should be considered when interpreting "
    "simulation outputs. These are not oversights — they are deliberate "
    "scope boundaries. A production-grade model would address each of these."
)

limitations = [
    (
        "No Ancillary Services Co-optimisation",
        "Energy and ancillary services are co-optimised in real ERCOT operations. "
        "A battery committed to RegUp cannot simultaneously discharge for energy arbitrage. "
        "Excluding ancillary services overstates energy arbitrage availability "
        "and understates total revenue."
    ),
    (
        "Price Taker at All Scales",
        "The price taker assumption breaks down for large batteries (>200 MW) "
        "or during hours of low liquidity. A large battery discharging at peak "
        "would suppress the very prices it is trying to capture."
    ),
    (
        "No Start-Up or No-Load Costs",
        "Unlike thermal generators, batteries have no start-up costs. "
        "However, very frequent cycling accelerates degradation — "
        "this cost is not captured in the revenue figures."
    ),
    (
        "DAM Only — No RTM Participation",
        "ERCOT batteries can earn additional revenue by deviating from "
        "their DAM schedule in real time when RTM prices spike. "
        "This optionality is excluded from the model."
    ),
    (
        "Basis Risk",
        "Settlement at LZ_HOUSTON_DAM may differ from the battery's "
        "actual settlement node price. Congestion within the Houston zone "
        "can create meaningful basis between zonal and nodal prices."
    ),
    (
        "No O&M Costs",
        "Operations and maintenance costs (typically $5–15/MWh) "
        "are not deducted from revenue figures."
    ),
]

for title, description in limitations:
    with st.expander(f"⚠️ {title}"):
        st.markdown(description)

# ── Section 7: Data Sources ────────────────────────────────────────────────
st.divider()
st.subheader("📂 Data Sources")

st.markdown("""
| Data Type | Source | Frequency | Notes |
|---|---|---|---|
| DAM Zonal Prices | ERCOT MIS | Hourly | LZ_HOUSTON, LZ_SOUTH, LZ_WEST, LZ_NORTH, LZ_RAYBN |
| Regional Load | ERCOT MIS | Hourly | 5 load zones |
| Wind Generation | ERCOT MIS / GridStatus | Hourly | Regional + system total |
| Solar Generation | ERCOT MIS / GridStatus | Hourly | Regional + system total |
| Weather Data | NOAA / preprocessed | Hourly | Temperature, humidity, heat index per zone |
| Calendar Features | Derived | Hourly | Hour, month, year, weekend, holiday flags |
""")

st.caption(
    "All data has been preprocessed and validated. "
    "See the Data Quality Assurance page for coverage statistics, "
    "missing value summaries, and duplicate timestamp checks."
)
