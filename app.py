import streamlit as st

st.set_page_config(
    page_title="Application",
    layout="wide",
)


def main() -> None:
    st.title("Application")
    st.subheader(
        "Explore ERCOT DAM price spreads, their drivers, and battery arbitrage opportunities."
    )

    st.markdown("### What you can do in this app")
    st.markdown(
        """
- **Dashboard — DAM Price Spreads**: High-level view of zonal DAM spreads across ERCOT load zones.
- **Spread Drivers**: Inspect which fundamentals (load, weather, renewables) explain specific spreads.
- **Battery Simulation**: Sketch how a grid-scale battery could charge and discharge across spreads.
- **Data QA**: Review basic coverage and sanity checks on the underlying datasets.
        """
    )

    with st.sidebar:
        st.markdown("## Application")
        st.caption("ERCOT battery trading playground")
        st.markdown("---")
        st.markdown(
            "Use the **page selector** in the sidebar to navigate between "
            "Dashboard, Spread Drivers, Battery Simulation, and Data QA."
        )


if __name__ == "__main__":
    main()

