from __future__ import annotations

import pandas as pd


def date_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of date coverage per region loaded.
    Shows start date, end date, total hours, expected hours,
    and coverage percentage.
    """
    from src.data import load_all_regions

    regions = load_all_regions()
    rows = []

    for region, rdf in regions.items():
        start = rdf.index.min()
        end = rdf.index.max()
        total_hours = len(rdf)
        expected_hours = int((end - start).total_seconds() / 3600) + 1
        coverage_pct = min(round(total_hours / expected_hours * 100, 2), 100.0) if expected_hours > 0 else 0.0

        rows.append({
            "Region": region.capitalize(),
            "Start": start.strftime("%Y-%m-%d"),
            "End": end.strftime("%Y-%m-%d"),
            "Total Hours": total_hours,
            "Expected Hours": expected_hours,
            "Coverage (%)": coverage_pct,
            "Status": "✅ Good" if coverage_pct >= 98 else "⚠️ Gaps Found",
        })

    return pd.DataFrame(rows)


def missing_values_summary(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    """
    Returns a per-column summary of missing values for a regional DataFrame.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = round(null_count / n * 100, 2) if n > 0 else 0.0
        rows.append({
            "Column": col,
            "Missing": null_count,
            "Missing (%)": null_pct,
            "Status": "✅" if null_pct == 0 else ("⚠️" if null_pct < 5 else "❌"),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Missing (%)", ascending=False)
        .reset_index(drop=True)
    )


def duplicate_timestamps(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    """
    Detects duplicate datetime index entries.
    Returns the duplicate rows or an empty DataFrame if none found.
    """
    dupes = df[df.index.duplicated(keep=False)].copy()
    if dupes.empty:
        return dupes
    dupes.index.name = "Datetime"
    dupes = dupes.reset_index()
    dupes.insert(0, "Region", region_name.capitalize())
    return dupes
