# BatteryArbitrage-main — ERCOT Battery Trading & Market Analytics

Streamlit app for ERCOT Day-Ahead Market (DAM) price spreads, spread drivers, and grid-scale battery arbitrage. The battery simulation uses **forecast-driven DAM bidding**: an XGBoost model predicts Houston DAM prices; bids are optimised on those forecasts and settled at actual LZ_HOUSTON_DAM.

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
