import streamlit as st

st.set_page_config(
    page_title="Application",
    layout="wide",
)


def main() -> None:
    st.title("ERCOT Battery Trading & Market Analytics")
    st.subheader(
        "Understanding battery economics through forecast-driven DAM bidding simulation"
    )

    st.markdown("""
    ### The Problem
    
    Grid-scale batteries in ERCOT must commit to day-ahead schedules before knowing actual prices. 
    This tool helps answer: **How do forecast errors impact battery revenue?** and **What drives 
    profitable arbitrage opportunities?**
    """)

    st.markdown("### What you can do in this app")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Dashboard**
        
        Explore DAM price spreads across ERCOT load zones. Identify when and where 
        arbitrage opportunities exist through time series analysis, histograms, and 
        opportunity tables.
        
        **🔍 Spread Drivers**
        
        Train XGBoost models to predict prices. Understand which fundamentals 
        (load, weather, renewables) drive spreads and evaluate forecast accuracy.
        """)
    
    with col2:
        st.markdown("""
        **🔋 Battery Simulation**
        
        Configure a Houston BESS and simulate forecast-driven DAM bidding. Compare 
        forecast-driven revenue to perfect-foresight benchmarks to quantify the cost 
        of forecast uncertainty.
        
        **✅ Data QA**
        
        Review data coverage, missing values, and quality checks across regional datasets.
        """)

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

