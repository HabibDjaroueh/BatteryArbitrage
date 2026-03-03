"""
Merged test suite from test_1_5.ipynb and test_6_12.ipynb.
Run from project root: python test.py  or  python -m test
"""
import datetime
import io
import pandas as pd
import plotly.graph_objects as go

from src.data import load_region_df, load_all_regions, get_lz_col, get_load_col
from src.filters import apply_filters
from src.kpis import compute_kpis
from src.charts import (
    net_load_vs_spread,
    renewables_vs_spread,
    net_load_time_series,
    net_load_duck_curve,
    net_load_vs_price,
)
from src.tables import build_opportunity_table


def check(condition, msg_pass, msg_fail):
    global passed, failed
    if condition:
        print(f"✅ {msg_pass}")
        passed += 1
    else:
        print(f"❌ {msg_fail}")
        failed += 1


passed = 0
failed = 0

# ─── Part 1: Data loading (from test_1_5) ───────────────────────────────────
print("\n--- Part 1: Data & region loading ---")
df = load_region_df("coast")
assert df.index.name == "datetime", "❌ datetime not set as index"
assert str(df.index.dtype) == "datetime64[ns]", "❌ index not parsed as datetime"
assert df.shape[0] > 0, "❌ DataFrame is empty"
print("✅ load_region_df('coast') passed")

non_float = [c for c in df.columns if df[c].dtype not in ["float64", "float32"]]
assert len(non_float) == 0, f"❌ Non-float columns found: {non_float}"
print("✅ All columns are float dtype")

all_dfs = load_all_regions()
assert set(all_dfs.keys()) == {"coast", "south", "west", "north", "east"}
print("✅ load_all_regions() passed")

assert get_lz_col("coast") == "LZ_HOUSTON_DAM"
assert get_lz_col("east") == "LZ_RAYBN_DAM"
assert get_load_col("west") == "WEST_Load"
print("✅ get_lz_col / get_load_col passed")

try:
    load_region_df("invalid")
    print("❌ Should have raised ValueError")
    failed += 1
except ValueError:
    print("✅ ValueError raised for invalid region")
    passed += 1

# ─── Part 2: KPIs (adapted to compute_kpis(df, spread_col)) ───────────────────
print("\n--- Part 2: KPIs ---")
lz_col = get_lz_col("coast")
load_col = get_load_col("coast")
spread_col = "spread_h_s"
kpis = compute_kpis(df, spread_col)

assert isinstance(kpis, dict), "❌ compute_kpis must return a dict"
assert len(kpis) == 5, "❌ Expected 5 KPIs"
expected_keys = ["avg_spread", "volatility", "capture_rate", "re_curtailment", "net_load_stress"]
for k in expected_keys:
    assert k in kpis, f"❌ Missing {k}"
print("✅ Test — all 5 KPIs present")
print(f"   {kpis}")
passed += 1

for name, val in kpis.items():
    assert isinstance(val, str), f"❌ {name} value must be a string, got {type(val)}"
print("✅ Test — all KPI values are formatted strings")
passed += 1

controls = {
    "date_range": (datetime.date(2022, 6, 28), datetime.date(2025, 11, 9)),
    "hour_range": (0, 23),
    "day_type": "All",
    "season": "All",
    "spread": spread_col,
    "region": "coast",
}
filtered = apply_filters(df, controls)
kpis_full = compute_kpis(df, spread_col)
controls_summer = {**controls, "season": "Summer"}
df_summer = apply_filters(df, controls_summer)
kpis_summer = compute_kpis(df_summer, spread_col)
assert kpis_full["avg_spread"] != kpis_summer["avg_spread"]
print("✅ Test — KPIs change with filters")
print(f"   Full avg spread:   {kpis_full['avg_spread']}")
print(f"   Summer avg spread: {kpis_summer['avg_spread']}")
passed += 1

kpis_coast = compute_kpis(load_region_df("coast"), spread_col)
kpis_west = compute_kpis(load_region_df("west"), "spread_w_h")
assert kpis_coast["avg_spread"] != kpis_west["avg_spread"]
print("✅ Test — KPIs differ across regions")
passed += 1

controls_narrow = {
    "date_range": (datetime.date(2023, 7, 1), datetime.date(2023, 7, 31)),
    "hour_range": (8, 18),
    "day_type": "Weekday",
    "season": "Summer",
    "spread": spread_col,
    "region": "coast",
}
df_narrow = apply_filters(df, controls_narrow)
kpis_narrow = compute_kpis(df_narrow, spread_col)
print(f"✅ Test — KPIs compute on small filtered window ({len(df_narrow)} rows)")
passed += 1

# ─── Part 3: Charts (from test_6_12) ────────────────────────────────────────
print("\n--- Part 3: Charts ---")
check(
    spread_col in filtered.columns,
    "Test 1 — spread_h_s present in coast df",
    "Test 1 — spread_h_s NOT found in coast df",
)
check(
    filtered[spread_col].notna().sum() > 100,
    "Test 1 — spread_h_s has sufficient non-null values",
    "Test 1 — spread_h_s is mostly null",
)

hi_col = "COAST_Heat_Index"
hi_mean = filtered[hi_col].mean() if hi_col in filtered.columns else None
check(hi_col in filtered.columns, f"Test 2 — {hi_col} present", f"Test 2 — {hi_col} NOT found")
if hi_mean is not None:
    check(
        32 < hi_mean < 150,
        f"Test 2 — heat index mean {hi_mean:.1f} is in °F range",
        f"Test 2 — heat index mean {hi_mean:.1f} looks wrong — expected °F (32–150)",
    )

wind_cols = [c for c in filtered.columns if "wind" in c.lower()]
solar_cols = [c for c in filtered.columns if "solar" in c.lower()]
check(len(wind_cols) > 0, f"Test 3 — wind columns found: {wind_cols}", "Test 3 — no wind columns found")
check(len(solar_cols) > 0, f"Test 3 — solar columns found: {solar_cols}", "Test 3 — no solar columns found")

try:
    fig4 = net_load_vs_spread(filtered, spread_col, load_col, "coast")
    trace_names = [t.name for t in fig4.data]
    check(isinstance(fig4, go.Figure), "Test 4 — returns go.Figure", "Test 4 — did not return go.Figure")
    check(len(fig4.data) >= 2, "Test 4 — has >= 2 traces", "Test 4 — fewer than 2 traces")
    check(
        any("Heat Index" in (n or "") for n in trace_names),
        f"Test 4 — heat index trace present: {trace_names}",
        f"Test 4 — heat index trace MISSING: {trace_names}",
    )
    yaxis2_title = fig4.layout.yaxis2.title.text if fig4.layout.yaxis2 else ""
    check(
        "°F" in (yaxis2_title or ""),
        f"Test 4 — secondary y-axis shows °F: '{yaxis2_title}'",
        f"Test 4 — secondary y-axis shows '{yaxis2_title}' — expected °F",
    )
except Exception as e:
    print(f"❌ Test 4 — net_load_vs_spread() crashed: {e}")
    failed += 1

try:
    fig5 = renewables_vs_spread(filtered, spread_col, "coast")
    trace_names = [t.name for t in fig5.data]
    check(isinstance(fig5, go.Figure), "Test 5 — returns go.Figure", "Test 5 — did not return go.Figure")
    check(any("Wind" in (n or "") for n in trace_names), "Test 5 — Wind trace present", f"Test 5 — Wind trace MISSING: {trace_names}")
    check(any("Solar" in (n or "") for n in trace_names), "Test 5 — Solar trace present", f"Test 5 — Solar trace MISSING: {trace_names}")
    check(not any("Heat Index" in (n or "") for n in trace_names), "Test 5 — heat index bubble removed", f"Test 5 — heat index bubble still present: {trace_names}")
except Exception as e:
    print(f"❌ Test 5 — renewables_vs_spread() crashed: {e}")
    failed += 1

try:
    fig6 = net_load_time_series(filtered, load_col, spread_col)
    trace_names = [t.name for t in fig6.data]
    raw_load_mean = filtered[load_col].resample("D").mean().mean()
    net_load_trace = next((t for t in fig6.data if "Net Load" in (t.name or "")), None)
    net_load_mean = pd.Series(net_load_trace.y).mean() if net_load_trace else None
    check(isinstance(fig6, go.Figure), "Test 6 — returns go.Figure", "Test 6 — did not return go.Figure")
    check(any("Net Load" in (n or "") for n in trace_names), "Test 6 — Net Load trace present", f"Test 6 — Net Load trace MISSING: {trace_names}")
    check(any("Spread" in (n or "") for n in trace_names), "Test 6 — Spread trace present", f"Test 6 — Spread trace MISSING: {trace_names}")
    if net_load_mean is not None:
        check(
            net_load_mean < raw_load_mean,
            f"Test 6 — net load ({net_load_mean:,.0f} MW) < raw load ({raw_load_mean:,.0f} MW)",
            f"Test 6 — net load ({net_load_mean:,.0f} MW) >= raw load ({raw_load_mean:,.0f} MW) — renewables not subtracted",
        )
except Exception as e:
    print(f"❌ Test 6 — net_load_time_series() crashed: {e}")
    failed += 1

try:
    fig7 = net_load_duck_curve(filtered, load_col)
    years_in_data = sorted(filtered.index.year.unique())
    check(isinstance(fig7, go.Figure), "Test 7 — returns go.Figure", "Test 7 — did not return go.Figure")
    check(len(fig7.data) == len(years_in_data), f"Test 7 — one trace per year ({len(years_in_data)} years)", f"Test 7 — expected {len(years_in_data)} traces, got {len(fig7.data)}")
    check(all(len(t.x) == 24 for t in fig7.data), "Test 7 — each trace has 24 hourly points", "Test 7 — some traces don't have 24 points")
except Exception as e:
    print(f"❌ Test 7 — net_load_duck_curve() crashed: {e}")
    failed += 1

try:
    fig8 = net_load_vs_price(filtered, load_col, spread_col)
    scatter_traces = [t for t in fig8.data if isinstance(t, go.Scatter) and t.mode and "markers" in t.mode]
    check(isinstance(fig8, go.Figure), "Test 8 — returns go.Figure", "Test 8 — did not return go.Figure")
    check(len(scatter_traces) > 0, "Test 8 — marker trace present", "Test 8 — no marker trace found")
    if scatter_traces:
        check(scatter_traces[0].marker.colorscale is not None, "Test 8 — colorscale present on markers", "Test 8 — colorscale missing")
        check(scatter_traces[0].marker.showscale is True, "Test 8 — colorbar visible", "Test 8 — colorbar not showing")
except Exception as e:
    print(f"❌ Test 8 — net_load_vs_price() crashed: {e}")
    failed += 1

try:
    controls_summer = {
        "date_range": (datetime.date(2023, 1, 1), datetime.date(2025, 11, 9)),
        "hour_range": (6, 22),
        "day_type": "Weekday",
        "season": "Summer",
        "spread": spread_col,
        "region": "coast",
    }
    df_summer = apply_filters(df, controls_summer)
    check(len(df_summer) > 0, f"Test 9 — summer weekday filter returns {len(df_summer)} rows", "Test 9 — summer weekday filter returned empty df")
    net_load_vs_spread(df_summer, spread_col, load_col, "coast")
    renewables_vs_spread(df_summer, spread_col, "coast")
    net_load_time_series(df_summer, load_col, spread_col)
    net_load_duck_curve(df_summer, load_col)
    net_load_vs_price(df_summer, load_col, spread_col)
    check(True, "Test 9 — all 5 charts render on Summer Weekday filter", "")
except Exception as e:
    print(f"❌ Test 9 — crashed on summer weekday filter: {e}")
    failed += 1

try:
    from src.charts import lz_price_comparison  # noqa: F401
    check(False, "", "Test 10 — lz_price_comparison still importable — should be deleted")
except ImportError:
    check(True, "Test 10 — lz_price_comparison correctly removed", "")

# ─── Part 4: Opportunity table (from test_6_12) ───────────────────────────────
print("\n--- Part 4: Opportunity table ---")
opp_df = None
region = "coast"
try:
    opp_df = build_opportunity_table(filtered, spread_col, load_col, region)
    check(
        isinstance(opp_df, pd.DataFrame),
        "Test 1 — build_opportunity_table() returns a DataFrame",
        "Test 1 — did not return a DataFrame",
    )
except Exception as e:
    print(f"❌ Test 1 — build_opportunity_table() crashed: {e}")
    failed += 1

if opp_df is not None:
    expected_cols = [
        "Datetime",
        "Spread ($/MWh)",
        "Direction",
        "Hour",
        "Wind Gen (MW)",
        "Net Load (%)",
        "Heat Index (°F)",
    ]
    for col in expected_cols:
        check(
            col in opp_df.columns,
            f"Test 2 — column '{col}' present",
            f"Test 2 — column '{col}' MISSING — found: {list(opp_df.columns)}",
        )

    abs_spreads = opp_df["Spread ($/MWh)"].abs().values
    check(
        all(abs_spreads[i] >= abs_spreads[i + 1] for i in range(len(abs_spreads) - 1)),
        "Test 3 — rows sorted by absolute spread descending",
        "Test 3 — rows NOT correctly sorted by absolute spread",
    )

    source_max_abs = filtered[spread_col].abs().max()
    table_max_abs = opp_df["Spread ($/MWh)"].abs().max()
    check(
        abs(source_max_abs - table_max_abs) < 0.1,
        f"Test 4 — max abs spread matches source: ${table_max_abs:.2f}",
        f"Test 4 — max abs spread mismatch: table ${table_max_abs:.2f} vs source ${source_max_abs:.2f}",
    )

    hour_vals = opp_df["Hour"].dropna()
    check(
        hour_vals.min() >= 0 and hour_vals.max() <= 23,
        "Test 5 — Hour in range 0–23",
        "Test 5 — Hour out of range",
    )

    net_load_pct = opp_df["Net Load (%)"].dropna()
    if len(net_load_pct) > 0:
        check(
            net_load_pct.min() >= 0 and net_load_pct.max() <= 100,
            f"Test 6 — Net Load (%) in range 0–100% (min {net_load_pct.min():.1f}%, max {net_load_pct.max():.1f}%)",
            "Test 6 — Net Load (%) out of range",
        )
    else:
        print("⚠️  Test 6 — Net Load (%) all null — skipped")

    valid_directions = {"▲ Premium", "▼ Discount", "— Flat"}
    unique_directions = set(opp_df["Direction"].unique())
    check(
        unique_directions.issubset(valid_directions),
        f"Test 10 — direction labels valid: {unique_directions}",
        f"Test 10 — unexpected direction labels: {unique_directions - valid_directions}",
    )
    premium_rows = opp_df[opp_df["Direction"] == "▲ Premium"]
    discount_rows = opp_df[opp_df["Direction"] == "▼ Discount"]
    check(
        (premium_rows["Spread ($/MWh)"] > 0).all(),
        "Test 10 — all Premium rows have positive spread",
        "Test 10 — Premium rows contain non-positive spread values",
    )
    check(
        (discount_rows["Spread ($/MWh)"] < 0).all(),
        "Test 10 — all Discount rows have negative spread",
        "Test 10 — Discount rows contain non-negative spread values",
    )

    sample = opp_df["Datetime"].iloc[0]
    check(
        isinstance(sample, str) and len(sample) >= 14,
        f"Test 11 — Datetime is formatted string: '{sample}'",
        f"Test 11 — Datetime format unexpected: '{sample}'",
    )

    for n in [10, 50, 100]:
        sliced = opp_df.head(n)
        check(
            len(sliced) == min(n, len(opp_df)),
            f"Test 12 — top {n} slice returns {len(sliced)} rows",
            f"Test 12 — top {n} slice returned {len(sliced)} rows",
        )

    premium_only = opp_df[opp_df["Direction"] == "▲ Premium"]
    discount_only = opp_df[opp_df["Direction"] == "▼ Discount"]
    check(len(premium_only) > 0, f"Test 13 — Premium filter returns {len(premium_only)} rows", "Test 13 — Premium filter returned 0 rows")
    check(len(discount_only) > 0, f"Test 13 — Discount filter returns {len(discount_only)} rows", "Test 13 — Discount filter returned 0 rows")
    check(
        len(premium_only) + len(discount_only) <= len(opp_df),
        "Test 13 — Premium + Discount rows <= total rows",
        "Test 13 — Premium + Discount rows exceed total rows",
    )

    try:
        csv_bytes = opp_df.head(50).to_csv(index=False).encode("utf-8")
        csv_back = pd.read_csv(io.BytesIO(csv_bytes))
        check(
            list(csv_back.columns) == list(opp_df.columns),
            f"Test 14 — CSV export round-trips correctly ({len(csv_back)} rows)",
            "Test 14 — CSV columns mismatch after round-trip",
        )
    except Exception as e:
        print(f"❌ Test 14 — CSV export failed: {e}")
        failed += 1

try:
    controls_narrow = {
        "date_range": (datetime.date(2023, 7, 1), datetime.date(2023, 7, 31)),
        "hour_range": (8, 18),
        "day_type": "Weekday",
        "season": "Summer",
        "spread": spread_col,
        "region": "coast",
    }
    df_narrow = apply_filters(df, controls_narrow)
    opp_narrow = build_opportunity_table(df_narrow, spread_col, load_col, "coast")
    check(
        len(opp_narrow) > 0,
        f"Test 15 — table builds on narrow filter ({len(opp_narrow)} rows)",
        "Test 15 — table returned 0 rows on narrow filter",
    )
except Exception as e:
    print(f"❌ Test 15 — crashed on narrow filter: {e}")
    failed += 1

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'─' * 50}")
print(f"Results: {passed} passed / {failed} failed / {passed + failed} total")
if failed == 0:
    print("🎉 All tests passed")
else:
    print("⚠️  Fix the failing tests before moving on")
print(f"{'─' * 50}")
