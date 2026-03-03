import streamlit as st


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
                    Model Assumptions
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
        "Scope boundaries and simplifications for the battery simulation. "
        "Use this page to see what the model covers and what it leaves out."
    )

    # Battery & physical setup
    st.divider()
    _section_header("Battery & Physical Setup")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **Location**  
                Battery is located in the Houston load zone (`LZ_HOUSTON_DAM`); all revenue settles at the Houston DAM zonal price. Nodal pricing and basis risk are not modelled.

                **Technology & capacity**  
                Lithium-ion BESS with user-defined power (MW) and duration (hours). Energy capacity = MW × hours; dispatch cannot exceed rated power in any hour.
                """
            )

        with col2:
            st.markdown(
                """
                **State of charge (SoC)**  
                SoC is bounded between 0% and 100%. The battery starts each simulation at 50% SoC and will not discharge below 10% to preserve a buffer.

                **Degradation & interconnection**  
                Capacity fade, O&M costs, transmission constraints and congestion are not modelled; the battery is assumed fully interconnected.
                """
            )

    # Market structure
    st.divider()
    _section_header("Market Structure")

    with st.container(border=True):
        col3, col4 = st.columns(2)

        with col3:
            st.markdown(
                """
                **Market**  
                Asset participates only in the ERCOT wholesale Day-Ahead Market (DAM). Bids for operating day D are submitted by 10:00 AM CPT on day D‑1.

                **Price taker**  
                The battery is a strict price taker; it cannot move clearing prices. This is reasonable below ~200 MW but optimistic for very large assets.
                """
            )

        with col4:
            st.markdown(
                """
                **Competition & settlement**  
                Markets are assumed perfectly competitive with no counterparty risk or bid/offer spreads. All dispatch clears and settles at `LZ_HOUSTON_DAM` zonal prices.

                **Merchant revenue**  
                Revenue is fully merchant; PPAs, tolling agreements and other contracts are not modelled, and transmission charges or congestion rents are excluded.
                """
            )

    # Revenue streams
    st.divider()
    _section_header("Revenue Streams")

    with st.container(border=True):
        st.markdown(
            """
            The simulation reports **energy arbitrage revenue only**. Other ERCOT revenue streams are listed for context.

            | Revenue Stream                         | Included | Notes                                     |
            |----------------------------------------|----------|-------------------------------------------|
            | DAM Energy Arbitrage                   | Yes      | Core model output                         |
            | Real-Time Energy Arbitrage             | No       | RTM participation not modelled            |
            | Regulation Up / Down, RRS, ECRS, etc. | No       | Major ancillary services, excluded        |
            | Non-Spinning Reserve                   | No       | Excluded                                  |
            | Capacity Payments                      | No       | ERCOT has no capacity market              |
            """
        )

        st.markdown(
            """
            **Revenue boundary**  
            Ancillary services typically contribute 40–60% of ERCOT battery revenue; a co-optimised battery could earn roughly **1.4x–1.7x** the energy-only results shown here.
            """
        )

    # Dispatch strategy
    st.divider()
    _section_header("Dispatch Strategy")

    with st.container(border=True):
        col5, col6 = st.columns(2)

        with col5:
            st.markdown(
                """
                **Forecast-driven DAM strategy**  
                An XGBoost model forecasts next-day hourly prices; the battery selects one charge and one discharge window per day based on these forecasts and is committed to the DAM schedule.
                """
            )

        with col6:
            st.markdown(
                """
                **Cycle limit and impact of error**  
                At most one full charge–discharge cycle per day is optimised. Forecast error directly reduces realised revenue and its cost is reported explicitly in the simulation outputs.
                """
            )

    # Forecast model
    st.divider()
    _section_header("Forecast Model")

    with st.container(border=True):
        col7, col8 = st.columns(2)

        with col7:
            st.markdown(
                """
                **Algorithm and features**  
                XGBoost gradient-boosted regression trained on historical hourly ERCOT data: regional load, wind and solar, weather and heat index, net load, renewable penetration and calendar features.

                **Training setup**  
                Temporal train/test split with the most recent 20% of the window held out. Training window, day type, season and hour filters are configured on the Drivers page.
                """
            )

        with col8:
            st.markdown(
                """
                **Feature lag and validation**  
                All features are lagged 24 hours so inputs for day D come from D‑1 actuals, eliminating lookahead bias. Performance metrics are MAE, RMSE and direction accuracy on the held-out test set.

                **Limitations and retraining**  
                The model cannot anticipate structural market changes or extreme events and should be retrained periodically; a model trained on 2022 data will understate renewable-driven price effects in 2025.
                """
            )

    # Known limitations
    st.divider()
    _section_header("Known Limitations")

    with st.container(border=True):
        st.markdown(
            "The assumptions below are deliberate scope boundaries that should be considered when interpreting results."
        )

        limitations = [
            (
                "No ancillary services co-optimisation",
                "Energy and ancillary services are not co-optimised; energy availability is overstated and total revenue is understated relative to a fully co-optimised stack.",
            ),
            (
                "Price taker at all scales",
                "For large batteries or thin trading hours, additional dispatch could move prices, breaking the price‑taker assumption.",
            ),
            (
                "No start-up, degradation or O&M costs",
                "Frequent cycling and ongoing O&M costs are not deducted, so long‑run net revenue is slightly higher than project finance cash flows.",
            ),
            (
                "DAM only — no RTM participation",
                "Real-time deviations from the DAM schedule and RTM optimisation are excluded, so optionality value is not captured.",
            ),
            (
                "Basis risk",
                "Settlement at `LZ_HOUSTON_DAM` may differ from nodal settlement prices; congestion within the zone can create significant basis.",
            ),
            (
                "Simplified physical constraints",
                "The SoC engine enforces energy and power limits but omits detailed inverter, thermal and AC‑side constraints.",
            ),
        ]

        for title, body in limitations:
            st.markdown(f"**{title}**  \n{body}\n")

    # Data sources
    st.divider()
    _section_header("Data Sources")

    with st.container(border=True):
        st.markdown(
            """
            | Data Type           | Source              | Frequency | Notes                                                   |
            |---------------------|---------------------|-----------|---------------------------------------------------------|
            | DAM zonal prices    | ERCOT MIS           | Hourly    | `LZ_HOUSTON`, `LZ_SOUTH`, `LZ_WEST`, `LZ_NORTH`, `LZ_RAYBN` |
            | Regional load       | ERCOT MIS           | Hourly    | Five ERCOT load zones                                   |
            | Wind generation     | ERCOT MIS / GridStatus | Hourly | Regional and system totals                              |
            | Solar generation    | ERCOT MIS / GridStatus | Hourly | Regional and system totals                              |
            | Weather data        | NOAA / preprocessed | Hourly    | Temperature, humidity, heat index per zone              |
            | Calendar features   | Derived             | Hourly    | Hour, month, year, weekend and holiday flags            |
            """
        )

        st.caption(
            "All inputs are preprocessed and validated; see the Data Quality Assurance page for coverage, missing values and duplicate timestamp checks."
        )


if __name__ == "__main__":
    main()
