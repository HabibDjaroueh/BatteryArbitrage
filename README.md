# BatteryArbitrage-main — ERCOT Battery Trading & Market Analytics

Streamlit app for ERCOT Day-Ahead Market (DAM) price spreads, spread drivers, and grid-scale battery arbitrage. The battery simulation uses **forecast-driven DAM bidding**: an XGBoost model predicts Houston DAM prices; bids are optimised on those forecasts and settled at actual LZ_HOUSTON_DAM.

---

## Problem Statement

**Why this matters:** Grid-scale battery storage is one of the fastest-growing asset classes in ERCOT, with over 5 GW of capacity added in 2023 alone. However, battery economics are complex and depend critically on:

1. **Price forecasting accuracy** — Batteries must commit to day-ahead schedules before knowing actual prices
2. **Arbitrage opportunity identification** — Understanding when and where spreads create profitable charge/discharge windows
3. **Forecast error cost** — Quantifying the revenue impact of imperfect price predictions

This tool addresses a real problem faced by battery developers, traders, and asset owners: **How do you evaluate battery economics when revenue depends on forecasts you don't yet have?**

**What we built:** An interactive dashboard that lets users:
- Explore ERCOT DAM price spreads across load zones to identify arbitrage opportunities
- Train and inspect price forecasting models to understand what drives prices
- Simulate battery dispatch under realistic forecast-driven bidding constraints
- Quantify the cost of forecast uncertainty by comparing forecast-driven revenue to perfect-foresight benchmarks

**Why ERCOT:** ERCOT is an ideal market for this analysis because:
- High price volatility creates significant arbitrage opportunities
- Growing renewable penetration increases price volatility and battery value
- Zonal pricing (vs nodal) simplifies the analysis while maintaining market realism
- Public data availability enables reproducible analysis

---

## Scoping Decisions

Given the 2-4 hour time constraint, I made deliberate choices about what to include and exclude:

### ✅ What's Included

- **DAM-only energy arbitrage** — Focused on the primary revenue stream rather than trying to model all ancillary services
- **Single location (Houston)** — Chose one load zone to enable deep analysis rather than shallow multi-zone coverage
- **Forecast-driven bidding** — Implemented realistic day-ahead constraints (24h forecast lag) rather than perfect-foresight optimization
- **Interactive exploration** — Built a dashboard users can interact with rather than a static report
- **Transparent assumptions** — Documented all model boundaries so users understand limitations

### ❌ What's Excluded (and why)

- **Ancillary services** — While 40-60% of battery revenue, co-optimization adds significant complexity. Energy arbitrage provides a solid foundation.
- **Real-time market participation** — RTM adds optionality but requires modeling deviations and settlement differences. DAM-only keeps scope manageable.
- **Multi-zone optimization** — Single location enables deeper analysis of forecast quality and error costs.
- **Battery degradation** — Important for long-term economics but adds complexity without changing core insights about forecast-driven dispatch.

**Rationale:** The goal was to build something **defensible and complete** within scope, not something that tries to do everything. A focused tool that clearly communicates its boundaries is more valuable than an ambitious tool with hidden assumptions.

---

## Features

| Page | Description |
|------|-------------|
| **Dashboard** | DAM price spreads by load zone, time series and histograms, net load and renewables views, opportunity table. |
| **Drivers** | Train an XGBoost model on load/weather/renewables; inspect feature importance and prediction vs actuals. |
| **Assumptions** | Documents model scope, forecast lag (24h), and battery dispatch strategy. |
| **Battery Simulation** | Configure a Houston BESS (power, duration, efficiency), train the Houston price model, run a DAM campaign with SoC continuity, and compare to a perfect-foresight benchmark (forecast error cost, capture ratio). |
| **Data QA** | Date coverage, missing values, and duplicate timestamps by region. |

---

## Setup

### Prerequisites

- **Python** 3.10+
- **Dependencies** (see `requirements.txt`): `streamlit`, `pandas`, `numpy`, `plotly`, `xgboost`, `scikit-learn`

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd BatteryArbitrage-main
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Upgrade pip and install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Alternatively, you can use the provided setup script:
   ```bash
   ./setup_venv.sh
   ```

---

## Run the app

Make sure your virtual environment is activated, then:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`). Use the sidebar to switch between pages.

To deactivate the virtual environment when you're done:
```bash
deactivate
```

---

## Data

- Regional CSVs live under **`data/regional_data/`** (e.g. `coast.csv`, `south.csv`, `west.csv`, `north.csv`, `east.csv`).
- Each file should have a **`datetime`** index and columns such as load, wind, solar, LZ DAM prices, weather, and Houston/spread columns used by the app and models.
- Paths are resolved from the **project root** so the app and tests work from any working directory.

---

## Project layout

```
BatteryArbitrage-main/
├── app.py                 # Streamlit entry point
├── requirements.txt
├── setup_venv.sh          # Script to initialize virtual environment
├── .gitignore
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Drivers.py
│   ├── 3_Assumptions.py
│   ├── 4_Battery_Simulation.py
│   └── 5_Data_Quality_Assurance.py
├── src/
│   ├── data.py            # Load regional DataFrames, paths
│   ├── controls.py        # Sidebar / region selector
│   ├── filters.py         # Date and filter helpers
│   ├── kpis.py            # Spread KPIs, premium/discount zones
│   ├── charts.py          # Plotly charts (spreads, net load, etc.)
│   ├── tables.py          # Opportunity table
│   ├── models.py          # XGBoost training, prepare_features
│   ├── qa.py              # Date coverage, missing values, duplicates
│   ├── soc_engine.py      # Battery physics (SoC, dispatch, clipping)
│   ├── forecaster.py      # Houston DAM price model, day-ahead forecasts
│   ├── dam_bidder.py      # DAM bid optimisation and settlement
│   ├── forecast_error.py # Perfect foresight vs DAM, error cost, capture ratio
│   ├── battery.py         # Legacy (deprecated)
│   └── battery_charts.py  # Legacy (deprecated)
├── data/
│   └── regional_data/     # coast.csv, south.csv, ...
└── tests/
    ├── test_soc_continuity.ipynb  # SoC continuity, B6 feasibility, B10 PF≥DAM
    ├── test_soc_engine.ipynb
    ├── test_forecaster.ipynb
    ├── test_forecaster_and_dam.ipynb
    ├── test_dam_bidder.ipynb
    └── test_forecast_error.ipynb
```

---

## Battery simulation (summary)

1. **Model** — XGBoost predicts 24h of **LZ_HOUSTON_DAM** using D-1 features (24h lag to avoid lookahead).
2. **Forecast** — Day-ahead Houston price forecast per operating day.
3. **Bid** — For each day, optimal charge/discharge windows are chosen from the forecast; schedule is capped to **achievable energy** from current SoC (no over-bid, so no clipping).
4. **Settlement** — Bids are settled at **actual** LZ_HOUSTON_DAM; a single SoC engine runs across days for continuity.
5. **Benchmark** — Perfect foresight (same physics and capping) gives an upper bound; **forecast error cost** = PF − DAM (reported as ≥ 0); **capture ratio** = DAM / PF.

---

## Testing

- **SoC continuity** — Run `tests/test_soc_continuity.ipynb` (or the root `test_soc_continuity.ipynb`) to check campaign columns, day-to-day SoC continuity, feasibility (B6), and PF ≥ DAM (B10). Run from project root so `src` and `data` resolve.
- **Other notebooks** in `tests/` cover the SoC engine, forecaster, DAM bidder, and forecast error logic.

---

## Development Approach & AI Usage

**AI tools used:** This project leveraged AI assistance (primarily Cursor/Claude) throughout development to accelerate implementation while maintaining code quality. AI was used for:

- **Code generation** — Initial scaffolding of Streamlit pages and data processing functions
- **Debugging** — Identifying and fixing issues in battery physics simulation and forecast error calculations
- **Documentation** — Generating clear explanations of ERCOT market mechanics and model assumptions
- **Code review** — Ensuring consistency across modules and catching edge cases

**Workflow:** AI was used as a pair programming partner — not to generate code blindly, but to:
1. Rapidly prototype components (dashboard layouts, chart configurations)
2. Refactor for clarity (extracting reusable functions, improving naming)
3. Validate energy market logic (ensuring ERCOT DAM rules were correctly implemented)
4. Write comprehensive documentation (assumptions page, README)

**Why this approach:** The assessment explicitly encourages AI usage. Rather than hiding it, this project demonstrates **effective AI collaboration** — using AI to move faster on implementation while maintaining deep understanding of the domain (ERCOT markets) and ensuring defensible analysis.

**Key insight:** AI excels at implementation speed, but domain knowledge (understanding ERCOT market structure, battery physics, forecast evaluation) required careful human oversight. The value is in the **combination** — AI accelerates execution, human expertise ensures correctness.
