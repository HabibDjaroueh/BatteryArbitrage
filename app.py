import streamlit as st

st.set_page_config(
    page_title="Application",
    layout="wide",
)


def main() -> None:
    # Inject dark terminal-style CSS
    st.markdown(
        """
        <style>
        /* Dark terminal theme */
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        
        /* Header bar */
        .terminal-header {
            background-color: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 1rem 2rem;
            margin: -1rem -1rem 2rem -1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .app-name {
            font-family: 'Courier New', monospace;
            font-size: 1.5rem;
            font-weight: 700;
            color: #58a6ff;
            letter-spacing: 0.05em;
        }
        
        .tagline {
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            color: #8b949e;
            font-weight: 400;
        }
        
        /* KPI tiles */
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .kpi-tile {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
        }
        
        .kpi-label {
            font-size: 0.75rem;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }
        
        .kpi-value {
            font-size: 1rem;
            color: #c9d1d9;
            font-weight: 600;
        }
        
        .kpi-note {
            font-size: 0.7rem;
            color: #58a6ff;
            margin-top: 0.25rem;
            font-style: italic;
        }
        
        /* Content sections */
        .content-section {
            margin: 2rem 0;
        }
        
        .section-title {
            font-family: 'Courier New', monospace;
            font-size: 1.1rem;
            color: #58a6ff;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
            border-bottom: 1px solid #30363d;
            padding-bottom: 0.5rem;
        }
        
        .bullet-list {
            list-style: none;
            padding-left: 0;
        }
        
        .bullet-list li {
            padding: 0.5rem 0;
            color: #c9d1d9;
            font-size: 0.95rem;
        }
        
        .bullet-list li:before {
            content: "▸ ";
            color: #58a6ff;
            font-weight: bold;
            margin-right: 0.5rem;
        }
        
        .module-item {
            padding: 0.75rem 0;
            border-bottom: 1px solid #21262d;
        }
        
        .module-item:last-child {
            border-bottom: none;
        }
        
        .module-name {
            font-family: 'Courier New', monospace;
            color: #58a6ff;
            font-weight: 600;
            font-size: 0.95rem;
        }
        
        .module-desc {
            color: #8b949e;
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }
        
        /* Call-to-action box */
        .cta-box {
            background-color: #1f6feb;
            border: 1px solid #388bfd;
            border-radius: 4px;
            padding: 1.5rem;
            margin: 2rem 0;
            text-align: center;
        }
        
        .cta-text {
            font-family: 'Courier New', monospace;
            font-size: 1.1rem;
            color: #ffffff;
            font-weight: 600;
        }
        
        /* Sidebar styling */
        .sidebar-content {
            font-family: 'Courier New', monospace;
        }
        
        .sidebar-title {
            font-size: 1.2rem;
            color: #58a6ff;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .sidebar-caption {
            color: #8b949e;
            font-size: 0.85rem;
        }
        
        .disclaimer-box {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-left: 3px solid #f85149;
            padding: 1rem;
            margin-top: 1.5rem;
            border-radius: 4px;
        }
        
        .disclaimer-title {
            font-size: 0.85rem;
            color: #f85149;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .disclaimer-text {
            font-size: 0.8rem;
            color: #8b949e;
            line-height: 1.5;
        }
        
        /* Override Streamlit defaults */
        h1, h2, h3 {
            color: #c9d1d9 !important;
        }
        
        .stMarkdown {
            color: #c9d1d9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header bar
    st.markdown(
        """
        <div class="terminal-header">
            <div class="app-name">ERCOT BATTERY ANALYTICS</div>
            <div class="tagline">Forecast-driven DAM bidding simulation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI tiles
    st.markdown(
        """
        <div class="kpi-container">
            <div class="kpi-tile">
                <div class="kpi-label">Market</div>
                <div class="kpi-value">ERCOT | DAM (Wholesale)</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Asset</div>
                <div class="kpi-value">100 MW / 200 MWh</div>
                <div class="kpi-note">Houston</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Horizon</div>
                <div class="kpi-value">2022–2025</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Mode</div>
                <div class="kpi-value">Energy Arbitrage</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two-column content section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="content-section">
                <div class="section-title">What this app answers</div>
                <ul class="bullet-list">
                    <li>How forecast errors impact battery revenue</li>
                    <li>What spread regimes create arbitrage opportunities</li>
                    <li>Which fundamentals drive price dynamics</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="content-section">
                <div class="section-title">Modules</div>
                <div class="module-item">
                    <div class="module-name">Dashboard</div>
                    <div class="module-desc">Price spreads, time series, and opportunity analysis</div>
                </div>
                <div class="module-item">
                    <div class="module-name">Spread Drivers</div>
                    <div class="module-desc">XGBoost price forecasting and feature importance</div>
                </div>
                <div class="module-item">
                    <div class="module-name">Battery Simulation</div>
                    <div class="module-desc">Forecast-driven DAM bidding with SoC continuity</div>
                </div>
                <div class="module-item">
                    <div class="module-name">Data QA</div>
                    <div class="module-desc">Coverage, missing values, and quality metrics</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-content">
                <div class="sidebar-title">ERCOT ANALYTICS</div>
                <div class="sidebar-caption">Battery Trading & Market Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")


if __name__ == "__main__":
    main()
