import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── constants ──────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # ── Regional Load ──────────────────────────────────────────────────────
    "COAST_Load", "SOUTH_Load", "WEST_Load", "NORTH_Load", "EAST_Load", "ERCOT_Load",

    # ── Regional Net Load ──────────────────────────────────────────────────
    "COAST_Net_Load", "SOUTH_Net_Load", "WEST_Net_Load",
    "NORTH_Net_Load", "EAST_Net_Load",

    # ── Regional Wind Generation ───────────────────────────────────────────
    "wind_gen_coast_mw", "wind_gen_south_mw", "wind_gen_west_mw",
    "wind_gen_north_mw", "wind_gen_east_mw",

    # ── Panhandle Wind (West Texas) ────────────────────────────────────────
    "wind_gen_panhandle_mw",

    # ── Regional Solar Generation ──────────────────────────────────────────
    "solar_gen_coast_mw", "solar_gen_south_mw", "solar_gen_west_mw",
    "solar_gen_north_mw", "solar_gen_east_mw",

    # ── System-Wide Totals ─────────────────────────────────────────────────
    "wind_gen_total_mw", "solar_gen_total_mw",

    # ── System-Wide Renewable Penetration % ───────────────────────────────
    "renewable_penetration",

    # ── Weather ────────────────────────────────────────────────────────────
    "temperature_C", "temperature_S", "temperature_W", "temperature_N", "temperature_E",
    "relative_humidity_C", "relative_humidity_S", "relative_humidity_W",
    "relative_humidity_N", "relative_humidity_E",
    "dew_point_temperature_C", "dew_point_temperature_S", "dew_point_temperature_W",
    "dew_point_temperature_N", "dew_point_temperature_E",
    "altimeter_C", "altimeter_S", "altimeter_W", "altimeter_N", "altimeter_E",
    "visibility_C", "visibility_S", "visibility_W", "visibility_N", "visibility_E",

    # ── Heat Index (per region, °F) ────────────────────────────────────────
    "COAST_Heat_Index", "SOUTH_Heat_Index", "WEST_Heat_Index",
    "NORTH_Heat_Index", "EAST_Heat_Index",

    # ── Calendar ───────────────────────────────────────────────────────────
    "hour", "month", "weekend", "holiday", "Year",
]

from src.controls import AVAILABLE_SPREADS

SPREAD_COLS = AVAILABLE_SPREADS

LZ_COLS = [
    "LZ_HOUSTON_DAM", "LZ_NORTH_DAM", "LZ_SOUTH_DAM",
    "LZ_WEST_DAM",    "LZ_RAYBN_DAM",
]


# ── feature preparation ────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects available feature columns from the DataFrame.
    Adds calendar features from the datetime index if not already present.
    Excludes spread and LZ price columns to prevent leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Regional DataFrame with datetime index.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for model training.
    """
    df = df.copy()

    # Add calendar features from index if missing
    if "hour"  not in df.columns: df["hour"]  = df.index.hour
    if "month" not in df.columns: df["month"] = df.index.month
    if "Year"  not in df.columns: df["Year"]  = df.index.year

    # Select only columns that exist in df and are not target-like
    available = [
        c for c in FEATURE_COLS
        if c in df.columns
        and c not in SPREAD_COLS
        and c not in LZ_COLS
    ]

    return df[available].copy()


# ── model training ─────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    target_col: str,
) -> dict:
    """
    Trains an XGBoost model to predict the target column.

    Parameters
    ----------
    df         : Regional DataFrame with datetime index (already filtered by date etc. by caller).
    target_col : Spread or LZ price column to predict.

    Returns
    -------
    dict with keys:
        model         : trained XGBRegressor
        feature_names : list of feature column names used
        importance_df : pd.DataFrame — Feature and Importance columns
        mae           : float
        rmse          : float
        direction_acc : float | None — % spread sign predicted correctly
        n_train       : int
        n_test        : int
        target_col    : str
    """

    # ── prepare features and target ────────────────────────────────────────
    X = prepare_features(df)
    y = df[target_col]

    # Align on index and drop nulls
    data = pd.concat([X, y], axis=1).dropna()
    X    = data[X.columns]
    y    = data[target_col]

    if len(X) < 100:
        raise ValueError(
            f"Not enough data to train after filtering: {len(X)} rows. "
            "Try widening your date range or selecting 'All Available Data'."
        )

    # ── temporal train / test split — no shuffle ───────────────────────────
    split_idx       = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx],  X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx],  y.iloc[split_idx:]

    # ── train XGBoost ──────────────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae    = float(mean_absolute_error(y_test, y_pred))
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Direction accuracy — only meaningful for spread targets
    if target_col in SPREAD_COLS:
        direction_acc = float(
            np.mean(np.sign(y_pred) == np.sign(y_test.values)) * 100
        )
    else:
        direction_acc = None

    # ── feature importance ─────────────────────────────────────────────────
    importance_df = (
        pd.DataFrame({
            "Feature":    X.columns.tolist(),
            "Importance": model.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model":         model,
        "feature_names": X.columns.tolist(),
        "importance_df": importance_df,
        "mae":           mae,
        "rmse":          rmse,
        "direction_acc": direction_acc,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "target_col":    target_col,
    }


# ── auto summary ───────────────────────────────────────────────────────────

def generate_summary(
    importance_df: pd.DataFrame,
    target_col: str,
    mae: float,
    rmse: float,
    direction_acc: float | None,
) -> str:
    """
    Generates a plain-English trader-facing summary of model results.

    Parameters
    ----------
    importance_df  : DataFrame with Feature and Importance columns.
    target_col     : Column the model was trained to predict.
    mae            : Mean absolute error on test set.
    rmse           : Root mean squared error on test set.
    direction_acc  : % of spread sign predicted correctly, or None.

    Returns
    -------
    str — markdown-formatted summary string.
    """
    top3  = importance_df["Feature"].head(3).tolist()
    top10 = importance_df["Feature"].head(10).tolist()

    target_label = (
        target_col.replace("spread_", "")
                  .replace("_", " → ")
                  .upper()
        if "spread" in target_col
        else target_col.replace("_", " ")
    )

    summary = (
        f"**Model target:** {target_label}  \n"
        f"**Top drivers:** {top3[0]}, {top3[1]}, and {top3[2]} "
        f"account for the largest share of predictive power.  \n"
        f"**Performance:** MAE = ${mae:.2f}/MWh · RMSE = ${rmse:.2f}/MWh"
    )

    if direction_acc is not None:
        summary += (
            f" · Direction accuracy = {direction_acc:.1f}%  \n"
            f"**Interpretation:** The model correctly predicts the spread "
            f"direction (positive vs negative) {direction_acc:.1f}% of the time. "
        )
        if direction_acc >= 70:
            summary += (
                "This is commercially useful — the model has sufficient "
                "directional signal to inform battery charge/discharge decisions."
            )
        elif direction_acc >= 60:
            summary += (
                "This is moderately useful — the model has some directional "
                "signal but consider adding more features or widening the training window."
            )
        else:
            summary += (
                "Direction accuracy is low — treat spread sign predictions "
                "with caution and do not rely on them for dispatch decisions alone."
            )

    summary += (
        f"  \n**Full top 10 features:** "
        + ", ".join(top10)
    )

    return summary
