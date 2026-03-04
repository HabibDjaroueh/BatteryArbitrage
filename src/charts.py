from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import zone mappings for anomaly detection and similar days
from src.kpis import ZONE_LOAD_MAP, ZONE_NET_LOAD_MAP, ZONE_WIND_COL, ZONE_SOLAR_COL

REGION_HEAT_INDEX = {
    "coast": "COAST_Heat_Index",
    "south": "SOUTH_Heat_Index",
    "west": "WEST_Heat_Index",
    "north": "NORTH_Heat_Index",
    "east": "EAST_Heat_Index",
}

SPREAD_COLOR_MAP = {
    "spread_h_s": "#38bdf8",  # sky blue
    "spread_h_n": "#60a5fa",  # light blue
    "spread_h_w": "#93c5fd",  # softer blue
    "spread_h_r": "#7dd3fc",  # cyan-blue
    "spread_n_s": "#1d4ed8",  # deep blue
    "spread_n_w": "#2563eb",  # medium blue
    "spread_n_r": "#0ea5e9",  # bright cyan
    "spread_s_w": "#3b82f6",  # primary blue
    "spread_s_r": "#1e40af",  # navy blue
    "spread_w_r": "#38bdf8",  # sky blue
    "spread_r_n": "#6366f1",  # indigo-leaning blue
    "spread_r_w": "#4f46e5",  # indigo-deep blue
}

CHART_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "#0a0e17",
    "plot_bgcolor": "#111827",
    "font": dict(family="monospace", color="#94a3b8", size=11),
    "gridcolor": "#1e2d40",
    "zero_line_color": "#475569",
    "accent": "#58a6ff",
}

STANDARD_HEIGHT = 420


def _apply_base_layout(
    fig: go.Figure,
    title: str,
    height: int = STANDARD_HEIGHT,
    margin: dict | None = None,
) -> None:
    """Apply consistent base layout to any figure."""
    standard_margin = dict(l=60, r=20, t=80, b=40)
    fig.update_layout(
        title=title,
        template=CHART_THEME["template"],
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=margin or standard_margin,
        height=height,
    )
    fig.update_xaxes(gridcolor=CHART_THEME["gridcolor"])
    fig.update_yaxes(gridcolor=CHART_THEME["gridcolor"])


def _spread_label(spread_col: str) -> str:
    """Converts spread_h_s → H → S for clean legend labels."""
    parts = spread_col.replace("spread_", "").split("_")
    return " → ".join(p.upper() for p in parts)


def get_heat_index_col(region: str, df: pd.DataFrame) -> str | None:
    """Return heat index column name for a region if it exists in df."""
    col = REGION_HEAT_INDEX.get(region)
    return col if col and col in df.columns else None


def spread_time_series(
    df: pd.DataFrame,
    spread_cols: list[str],
    show_rolling: bool = True,
) -> go.Figure:
    """
    Plots daily mean spread for one or two spread columns on the same chart.
    Each spread is a distinct color from SPREAD_COLOR_MAP.
    Rolling mean shown as a dotted line in the same color if show_rolling=True.

    Parameters
    ----------
    df           : filtered regional DataFrame
    spread_cols  : list of 1 or 2 spread column names
    show_rolling : whether to show 30D rolling mean per spread
    """

    fig = go.Figure()
    is_overlay = len(spread_cols) == 2

    for idx, spread_col in enumerate(spread_cols[:2]):   # hard cap at 2
        if spread_col not in df.columns:
            continue

        color   = SPREAD_COLOR_MAP.get(spread_col, "#94a3b8")
        label   = _spread_label(spread_col)
        daily   = df[spread_col].resample("D").mean()
        rolling = daily.rolling(30).mean()

        # Case 1: Primary raw series (idx=0)
        # Case 3: Overlay raw series (idx=1, is_overlay=True)
        if is_overlay and idx == 1:
            # Overlay raw: dashed, thinner, lighter blue
            overlay_color = "#7ab8ff"  # lighter blue
            line_style = dict(color=overlay_color, width=1.0, dash="dash")
            line_opacity = 0.75
        else:
            # Primary raw: solid, normal width, normal color
            line_style = dict(color=color, width=1.5)
            line_opacity = 0.85

        # Daily spread line
        fig.add_trace(go.Scatter(
            x=daily.index,
            y=daily.values,
            name=label,
            mode="lines",
            line=line_style,
            opacity=line_opacity,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "%{x|%b %d %Y}<br>"
                "Spread: $%{y:.2f}/MWh<extra></extra>"
            )
        ))

        # Case 2: Primary 30D MA (idx=0)
        # Case 4: Overlay 30D MA (idx=1, is_overlay=True)
        if show_rolling:
            if is_overlay and idx == 1:
                # Overlay 30D MA: dashed, thinner, lighter blue, lower opacity
                rolling_color = overlay_color
                rolling_style = dict(color=rolling_color, width=1.5, dash="dash")
                rolling_opacity = 0.6
            else:
                # Primary 30D MA: wider, dashed, lighter blue, more prominent
                rolling_color = "#93C5FD"  # lighter blue for prominence
                rolling_style = dict(color=rolling_color, width=3.0, dash="dash")
                rolling_opacity = 0.9
            
            fig.add_trace(go.Scatter(
                x=rolling.index,
                y=rolling.values,
                name=f"{label} 30D MA",
                mode="lines",
                line=rolling_style,
                opacity=rolling_opacity,
                hovertemplate=(
                    f"<b>{label} 30D MA</b><br>"
                    "%{x|%b %d %Y}<br>"
                    "$%{y:.2f}/MWh<extra></extra>"
                )
            ))

    # Zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)

    title = (
        f"Spread Time Series — {_spread_label(spread_cols[0])}"
        if len(spread_cols) == 1
        else f"Spread Overlay — {_spread_label(spread_cols[0])} vs {_spread_label(spread_cols[1])}"
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        xaxis=dict(
            gridcolor="#1e2d40",
            rangeselector=dict(
                buttons=[
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=12, label="1Y", step="month", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            )
        ),
        yaxis=dict(gridcolor="#1e2d40", title="Spread ($/MWh)"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        margin=dict(l=60, r=20, t=80, b=40),
        height=500,
    )

    return fig


def spread_histogram(
    df: pd.DataFrame,
    spread_cols: list[str],
) -> go.Figure:
    """
    Plots the hourly spread distribution for one or two spreads overlaid
    on the same histogram. Each spread uses its matching color from
    SPREAD_COLOR_MAP. P50 shown for both spreads, P10/P90 only for primary.

    Parameters
    ----------
    df          : filtered regional DataFrame
    spread_cols : list of 1 or 2 spread column names
    """

    fig = go.Figure()
    is_overlay = len(spread_cols) == 2

    for i, spread_col in enumerate(spread_cols[:2]):   # hard cap at 2
        if spread_col not in df.columns:
            continue

        color  = SPREAD_COLOR_MAP.get(spread_col, "#94a3b8")
        label  = _spread_label(spread_col)
        values = df[spread_col].dropna()
        if values.empty:
            continue

        p10 = values.quantile(0.10)
        p50 = values.quantile(0.50)
        p90 = values.quantile(0.90)

        # Case 1: Primary histogram bars (i=0)
        # Case 3: Overlay histogram bars (i=1, is_overlay=True)
        if is_overlay and i == 1:
            # Overlay histogram: lighter blue, lower opacity, more subdued
            overlay_color = "#7ab8ff"  # lighter blue
            histogram_color = overlay_color
            histogram_opacity = 0.35  # lower opacity for overlay
        else:
            # Primary histogram: normal color, higher opacity for prominence
            histogram_color = color
            histogram_opacity = 0.7 if len(spread_cols) == 2 else 0.8  # higher opacity for primary
        
        # Case 2: Primary percentile lines (i=0)
        # Case 4: Overlay percentile lines (i=1, is_overlay=True)
        if is_overlay and i == 1:
            # Overlay percentile lines: thinner, lighter blue, dashed
            overlay_color = "#7ab8ff"  # lighter blue
            p50_line_style = dict(color=overlay_color, width=2.5, dash="dash")  # thicker for visibility
            p50_color = overlay_color
        else:
            # Primary percentile lines: wider, brighter, more prominent
            p50_line_style = dict(color="#58a6ff", width=3.0, dash="dash")  # bright blue, thick for visibility
            p50_color = "#58a6ff"  # bright blue for visibility

        # Histogram trace
        fig.add_trace(go.Histogram(
            x=values,
            name=label,
            nbinsx=80,
            marker_color=histogram_color,
            opacity=histogram_opacity,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Spread: $%{x:.2f}/MWh<br>"
                "Count: %{y}<extra></extra>"
            )
        ))

        # P50 line — shown for both spreads
        # Stagger annotation vertically so they don't overlap
        fig.add_shape(
            type="line",
            x0=p50, x1=p50, y0=0, y1=1, yref="paper",
            line=p50_line_style
        )
        annotation_color = p50_color  # Use p50_color which is already set correctly for both cases
        fig.add_annotation(
            x=p50,
            y=0.92 - (i * 0.15),   # second spread annotation sits lower
            yref="paper",
            text=f"<b>{label} P50: ${p50:.1f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor=annotation_color,
            arrowwidth=1,
            ax=45, ay=0,
            font=dict(color=annotation_color, size=10),
            bgcolor="#0a0e17",
            bordercolor=annotation_color,
            borderwidth=1,
            borderpad=3,
        )

        # P10 / P90 lines — primary spread only to avoid clutter
        if i == 0:
            for val, pct_label, y_anchor in [
                (p10, f"P10: ${p10:.1f}", 0.72),
                (p90, f"P90: ${p90:.1f}", 0.57),
            ]:
                fig.add_shape(
                    type="line",
                    x0=val, x1=val, y0=0, y1=1, yref="paper",
                    line=dict(color="#58a6ff", width=2.5, dash="dot")  # bright blue, thick for visibility
                )
                fig.add_annotation(
                    x=val,
                    y=y_anchor,
                    yref="paper",
                    text=f"<b>{pct_label}</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#58a6ff",  # bright blue for visibility
                    arrowwidth=1.5,  # thicker arrow
                    ax=45, ay=0,
                    font=dict(color="#58a6ff", size=10),  # bright blue for visibility
                    bgcolor="#0a0e17",
                    bordercolor="#58a6ff",  # bright blue for visibility
                    borderwidth=1.5,  # thicker border
                    borderpad=3,
                )

    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1)

    title = (
        f"Spread Distribution — {_spread_label(spread_cols[0])}"
        if len(spread_cols) == 1
        else f"Spread Distributions — {_spread_label(spread_cols[0])} vs {_spread_label(spread_cols[1])}"
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        barmode="overlay",
        xaxis=dict(gridcolor="#1e2d40", title="Spread ($/MWh)"),
        yaxis=dict(gridcolor="#1e2d40", title="Count"),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        margin=dict(l=60, r=20, t=80, b=40),
        height=500,
    )

    return fig


def net_load_vs_spread(
    df: pd.DataFrame,
    spread_col: str,
    load_col: str,
    region: str,
) -> go.Figure:
    """Binned relationship between regional load, spread, and heat index."""
    if df.empty or spread_col not in df.columns or load_col not in df.columns:
        return go.Figure()

    df = df.copy()
    hi_col = get_heat_index_col(region, df)
    if hi_col:
        df["heat_index"] = df[hi_col]

    cols = [spread_col, load_col]
    if "heat_index" in df.columns:
        cols.append("heat_index")

    plot_df = df[cols].dropna()
    if plot_df.empty:
        return go.Figure()

    plot_df["load_bin"] = pd.qcut(
        plot_df[load_col], q=20, duplicates="drop"
    )

    agg = {
        "avg_load": (load_col, "mean"),
        "avg_spread": (spread_col, "mean"),
        "std_spread": (spread_col, "std"),
    }
    if "heat_index" in plot_df.columns:
        agg["avg_hi"] = ("heat_index", "mean")

    binned = (
        plot_df.groupby("load_bin", observed=True)
        .agg(**agg)
        .reset_index(drop=True)
    )
    if binned.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Error band ±1 std
    upper = binned["avg_spread"] + binned["std_spread"]
    lower = binned["avg_spread"] - binned["std_spread"]
    fig.add_trace(
        go.Scatter(
            x=pd.concat([binned["avg_load"], binned["avg_load"].iloc[::-1]]),
            y=pd.concat([upper, lower.iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(56,189,248,0.08)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    # Avg spread per load bin
    fig.add_trace(
        go.Scatter(
            x=binned["avg_load"],
            y=binned["avg_spread"],
            mode="lines+markers",
            name="Avg Spread",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=6, color="#38bdf8"),
            hovertemplate="Load: %{x:,.0f} MW<br>Spread: $%{y:.2f}/MWh<extra></extra>",
        ),
        secondary_y=False,
    )

    # Heat index on secondary y-axis
    if "avg_hi" in binned.columns:
        fig.add_trace(
            go.Scatter(
                x=binned["avg_load"],
                y=binned["avg_hi"],
                mode="lines",
                name="Avg Heat Index (°F)",
                line=dict(color="#fb923c", width=1.5, dash="dot"),
                hovertemplate="Load: %{x:,.0f} MW<br>Heat Index: %{y:.1f}°F<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)

    title_label = spread_col.replace("spread_", "").replace("_", " \u2192 ").upper()

    fig.update_layout(
        title=f"Net Load + Heat Index vs Spread ({title_label})",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=60, t=60, b=40),
        height=420,
    )
    fig.update_xaxes(gridcolor="#1e2d40", title_text="Regional Load (MW)")
    fig.update_yaxes(
        gridcolor="#1e2d40",
        title_text="Avg Spread ($/MWh)",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Heat Index (°F)", secondary_y=True, showgrid=False
    )

    return fig


def renewables_vs_spread(
    df: pd.DataFrame,
    spread_col: str,
    region: str,
) -> go.Figure:
    """Relationship between wind/solar generation and spread (no heat index overlay)."""
    if df.empty or spread_col not in df.columns:
        return go.Figure()

    df = df.copy()

    wind_cols = [c for c in df.columns if "wind" in c.lower()]
    solar_cols = [c for c in df.columns if "solar" in c.lower()]
    wind_col = wind_cols[0] if wind_cols else None
    solar_col = solar_cols[0] if solar_cols else None

    fig = go.Figure()

    for gen_col, color, name in [
        (wind_col, "#3b82f6", "Wind"),  # Darker blue for Wind - still visible on dark background
        (solar_col, "#38bdf8", "Solar"),  # Sky blue/cyan for Solar - more distinct
    ]:
        if gen_col is None:
            continue

        plot_df = df[[spread_col, gen_col]].dropna().copy()
        if plot_df.empty:
            continue

        plot_df["bin"] = pd.qcut(
            plot_df[gen_col], q=20, duplicates="drop"
        )
        binned = (
            plot_df.groupby("bin", observed=True)
            .agg(
                avg_re=(gen_col, "mean"),
                avg_spread=(spread_col, "mean"),
            )
            .reset_index(drop=True)
        )
        if binned.empty:
            continue

        # Use same color as line for markers - minimalistic, no red/green
        fig.add_trace(
            go.Scatter(
                x=binned["avg_re"],
                y=binned["avg_spread"],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color, opacity=0.7),  # Same color as line, slightly transparent
                hovertemplate=(
                    f"{name}: " + "%{x:,.0f} MW<br>"
                    "Spread: $%{y:.2f}/MWh<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)

    title_label = spread_col.replace("spread_", "").replace("_", " \u2192 ").upper()

    fig.update_layout(
        title=f"Wind & Solar vs Spread ({title_label}) — sized by Heat Index",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#1e2d40", title="Generation (MW)"),
        yaxis=dict(gridcolor="#1e2d40", title="Avg Spread ($/MWh)"),
        hovermode="closest",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=60, t=60, b=40),
        height=420,
    )

    return fig


def net_load_time_series(
    df: pd.DataFrame,
    load_col: str,
    spread_col: str,
) -> go.Figure:
    """Daily net load vs spread over the filtered window."""
    if df.empty or load_col not in df.columns or spread_col not in df.columns:
        return go.Figure()

    wind_cols = [c for c in df.columns if "wind" in c.lower()]
    solar_cols = [c for c in df.columns if "solar" in c.lower()]
    wind_col = wind_cols[0] if wind_cols else None
    solar_col = solar_cols[0] if solar_cols else None

    df = df.copy()
    df["net_load"] = df[load_col].copy()
    if wind_col and wind_col in df.columns:
        df["net_load"] -= df[wind_col]
    if solar_col and solar_col in df.columns:
        df["net_load"] -= df[solar_col]

    daily_nl = df["net_load"].resample("D").mean()
    daily_spread = df[spread_col].resample("D").mean()
    if daily_nl.empty or daily_spread.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=daily_nl.index,
            y=daily_nl.values,
            name="Net Load (MW)",
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.15)",
            line=dict(color="#a78bfa", width=1.5),
            hovertemplate="%{x|%b %d %Y}<br>Net Load: %{y:,.0f} MW<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=daily_spread.index,
            y=daily_spread.values,
            name="Daily Spread",
            line=dict(color="#38bdf8", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %d %Y}<br>Spread: $%{y:.2f}/MWh<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)

    title_label = spread_col.replace("spread_", "").replace("_", " \u2192 ").upper()

    fig.update_layout(
        title=f"Net Load vs Spread ({title_label}) — Daily",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=60, t=60, b=40),
        height=380,
    )
    fig.update_xaxes(gridcolor="#1e2d40")
    fig.update_yaxes(
        title_text="Net Load (MW)", gridcolor="#1e2d40", secondary_y=False
    )
    fig.update_yaxes(
        title_text="Spread ($/MWh)", showgrid=False, secondary_y=True
    )

    return fig


def net_load_duck_curve(
    df: pd.DataFrame,
    load_col: str,
) -> go.Figure:
    """Average net load profile by hour of day, split by year."""
    if df.empty or load_col not in df.columns:
        return go.Figure()

    wind_cols = [c for c in df.columns if "wind" in c.lower()]
    solar_cols = [c for c in df.columns if "solar" in c.lower()]
    wind_col = wind_cols[0] if wind_cols else None
    solar_col = solar_cols[0] if solar_cols else None

    df = df.copy()
    df["net_load"] = df[load_col].copy()
    if wind_col and wind_col in df.columns:
        df["net_load"] -= df[wind_col]
    if solar_col and solar_col in df.columns:
        df["net_load"] -= df[solar_col]

    if not isinstance(df.index, pd.DatetimeIndex):
        return go.Figure()

    df["hour"] = df.index.hour
    df["year"] = df.index.year

    years = sorted(df["year"].unique())
    if not years:
        return go.Figure()

    colors = ["#38bdf8", "#34d399", "#fb923c", "#a78bfa", "#f472b6"]

    fig = go.Figure()

    for i, year in enumerate(years):
        yr_df = df[df["year"] == year]
        hourly = yr_df.groupby("hour")["net_load"].mean()

        fig.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly.values,
                name=str(year),
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=(
                    f"{year} — Hour " "%{x}:00<br>Avg Net Load: %{y:,.0f} MW<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Avg Net Load by Hour of Day — Duck Curve (by Year)",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        xaxis=dict(
            gridcolor="#1e2d40",
            title="Hour of Day",
            tickmode="linear",
            tick0=0,
            dtick=2,
        ),
        yaxis=dict(gridcolor="#1e2d40", title="Avg Net Load (MW)"),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=40),
        height=380,
    )

    return fig


def net_load_vs_price(
    df: pd.DataFrame,
    load_col: str,
    spread_col: str,
) -> go.Figure:
    """Price response curve: net load vs spread, colored by avg hour of day."""
    if df.empty or load_col not in df.columns or spread_col not in df.columns:
        return go.Figure()

    wind_cols = [c for c in df.columns if "wind" in c.lower()]
    solar_cols = [c for c in df.columns if "solar" in c.lower()]
    wind_col = wind_cols[0] if wind_cols else None
    solar_col = solar_cols[0] if solar_cols else None

    df = df.copy()
    df["net_load"] = df[load_col].copy()
    if wind_col and wind_col in df.columns:
        df["net_load"] -= df[wind_col]
    if solar_col and solar_col in df.columns:
        df["net_load"] -= df[solar_col]

    if not isinstance(df.index, pd.DatetimeIndex):
        return go.Figure()

    df["hour"] = df.index.hour
    plot_df = df[["net_load", spread_col, "hour"]].dropna()
    if plot_df.empty:
        return go.Figure()

    plot_df["nl_bin"] = pd.qcut(plot_df["net_load"], q=30, duplicates="drop")
    binned = (
        plot_df.groupby("nl_bin", observed=True)
        .agg(
            avg_nl=("net_load", "mean"),
            avg_spread=(spread_col, "mean"),
            std_spread=(spread_col, "std"),
            avg_hour=("hour", "mean"),
        )
        .reset_index(drop=True)
    )
    if binned.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pd.concat([binned["avg_nl"], binned["avg_nl"].iloc[::-1]]),
            y=pd.concat(
                [
                    binned["avg_spread"] + binned["std_spread"],
                    (binned["avg_spread"] - binned["std_spread"]).iloc[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(56,189,248,0.08)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=binned["avg_nl"],
            y=binned["avg_spread"],
            mode="lines+markers",
            name="Avg Spread",
            line=dict(color="#38bdf8", width=2),
            marker=dict(
                size=8,
                color=binned["avg_hour"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Avg Hour", thickness=12),
            ),
            hovertemplate=(
                "Net Load: %{x:,.0f} MW<br>"
                "Avg Spread: $%{y:.2f}/MWh<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)

    title_label = spread_col.replace("spread_", "").replace("_", " \u2192 ").upper()

    fig.update_layout(
        title=f"Net Load vs Spread ({title_label}) — Price Response Curve",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#1e2d40", title="Net Load (MW)"),
        yaxis=dict(gridcolor="#1e2d40", title="Avg Spread ($/MWh)"),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=60, b=40),
        height=380,
    )

    return fig


# ── Monthly & Day Analysis ─────────────────────────────────────────────────

def monthly_spread_heatmap(
    df: pd.DataFrame,
    spread_col: str,
    col_label: str = "Spread",
    spread_label: str | None = None,
) -> go.Figure:
    """
    Calendar heatmap showing daily average spreads by month and day of month.
    
    Parameters
    ----------
    df          : filtered regional DataFrame with datetime index
    spread_col  : column containing spread values
    col_label   : label for the column (e.g., "Spread" or "Price")
    spread_label: formatted spread label for title (e.g., "H → S")
    """
    if df.empty or spread_col not in df.columns:
        return go.Figure()
    
    # Resample hourly data to daily averages
    daily = df[spread_col].resample("D").mean().dropna()
    if daily.empty:
        return go.Figure()
    
    # Create pivot table: rows = Year-Month, cols = Day-of-Month
    pivot_df = daily.to_frame("spread")
    pivot_df["year_month"] = pivot_df.index.strftime("%Y-%m")
    pivot_df["day"] = pivot_df.index.day
    pivot = pivot_df.pivot_table(index="year_month", columns="day", values="spread")
    
    if pivot.empty:
        return go.Figure()
    
    # Use blue-based colorscale (Blues or Viridis) instead of RdYlGn
    # Center at zero for balanced visualization
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(d) for d in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Blues",  # Blue-based instead of red-green
        zmid=0,  # Center at zero
        colorbar=dict(title="$/MWh", thickness=12),
        hovertemplate="Month: %{y}<br>Day: %{x}<br>Spread: $%{z:.2f}/MWh<extra></extra>",
    ))
    
    # Apply dark theme styling
    title_suffix = f" — {spread_label}" if spread_label else ""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        height=max(300, len(pivot) * 22),  # Dynamic height based on months
        title=f"Monthly {col_label} Heatmap{title_suffix}",
        xaxis_title="Day of Month",
        yaxis_title="",
        yaxis_autorange="reversed",  # Show most recent months at top
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def monthly_summary_bars(
    df: pd.DataFrame,
    spread_col: str,
    col_label: str = "Spread",
    spread_label: str | None = None,
) -> go.Figure:
    """
    Monthly average spreads with capture rate overlay.
    
    Parameters
    ----------
    df          : filtered regional DataFrame with datetime index
    spread_col  : column containing spread values
    col_label   : label for the column (e.g., "Spread" or "Price")
    spread_label: formatted spread label for title (e.g., "H → S")
    """
    if df.empty or spread_col not in df.columns:
        return go.Figure()
    
    # Group by month and calculate metrics
    tmp = df[[spread_col]].copy()
    tmp["year_month"] = tmp.index.to_period("M").astype(str)
    
    monthly = tmp.groupby("year_month")[spread_col].agg(
        avg_spread="mean",
        favorable_hours=lambda x: (x > 0).sum(),
        total_hours="count",
    )
    monthly["capture_rate"] = (monthly["favorable_hours"] / monthly["total_hours"] * 100).round(1)
    monthly = monthly.reset_index()
    
    if monthly.empty:
        return go.Figure()
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Color bars based on positive/negative spread
    # Use blue for positive, grey for negative (terminal theme)
    colors = ["#58a6ff" if v > 0 else "#94a3b8" for v in monthly["avg_spread"]]
    
    # Add bar chart for average spread
    fig.add_trace(go.Bar(
        x=monthly["year_month"], 
        y=monthly["avg_spread"],
        name="Avg Spread", 
        marker_color=colors, 
        opacity=0.8,
        hovertemplate="%{x}<br>Avg: $%{y:.2f}/MWh<extra></extra>",
    ), secondary_y=False)
    
    # Add line chart for capture rate
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], 
        y=monthly["capture_rate"],
        name="Capture Rate (%)", 
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=5),
        hovertemplate="%{x}<br>Capture: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)
    
    # Styling
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1, secondary_y=False)
    title_suffix = f" — {spread_label}" if spread_label else ""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        height=420,
        xaxis=dict(tickangle=-45, gridcolor="#1e2d40"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=60, r=60, t=60, b=40),
    )
    fig.update_yaxes(
        title_text=f"Avg {col_label} ($/MWh)",
        gridcolor="#1e2d40",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Capture Rate (%)",
        showgrid=False,
        secondary_y=True,
    )
    
    return fig


def day_of_week_spread(
    df: pd.DataFrame,
    spread_col: str,
    col_label: str = "Spread",
    spread_label: str | None = None,
) -> go.Figure:
    """
    Distribution analysis by weekday vs weekend patterns.
    
    Parameters
    ----------
    df          : filtered regional DataFrame with datetime index
    spread_col  : column containing spread values
    col_label   : label for the column (e.g., "Spread" or "Price")
    spread_label: formatted spread label for title (e.g., "H → S")
    """
    if df.empty or spread_col not in df.columns:
        return go.Figure()
    
    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    tmp = df[[spread_col]].copy()
    tmp["dow"] = tmp.index.dayofweek
    
    # Colors: weekdays in blue, weekends in lighter blue/grey (terminal theme)
    colors = ["#58a6ff"] * 5 + ["#7ab8ff"] * 2  # Weekdays: bright blue, weekends: lighter blue
    
    fig = go.Figure()
    
    for i, day_name in enumerate(DOW_NAMES):
        day_data = tmp[tmp["dow"] == i][spread_col].dropna()
        if not day_data.empty:
            fig.add_trace(go.Box(
                y=day_data, 
                name=day_name,
                marker_color=colors[i],
                boxmean=True,  # Show mean line
                hovertemplate=f"{day_name}<br>%{{y:.2f}} $/MWh<extra></extra>",
            ))
    
    if len(fig.data) == 0:
        return go.Figure()
    
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
    
    title_suffix = f" — {spread_label}" if spread_label else ""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        height=420,
        title=f"{col_label} by Day of Week{title_suffix}",
        yaxis_title=f"{col_label} ($/MWh)",
        yaxis=dict(gridcolor="#1e2d40"),
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    
    return fig


def compute_monthly_stats(
    df: pd.DataFrame,
    spread_col: str,
    lz_col: str | None = None,
) -> pd.DataFrame:
    """
    Detailed monthly metrics in tabular format.
    
    Parameters
    ----------
    df         : filtered regional DataFrame with datetime index
    spread_col  : column containing spread values
    lz_col      : optional LZ price column for reference
    
    Returns
    -------
    pd.DataFrame with monthly statistics
    """
    if df.empty or spread_col not in df.columns:
        return pd.DataFrame()
    
    tmp = df.copy()
    tmp["year_month"] = tmp.index.to_period("M")
    
    # Calculate monthly aggregations
    agg = tmp.groupby("year_month").agg(
        avg_spread=(spread_col, "mean"),
        max_spread=(spread_col, "max"),
        min_spread=(spread_col, "min"),
        std_spread=(spread_col, "std"),
        favorable_hrs=(spread_col, lambda x: (x > 0).sum()),
        total_hrs=(spread_col, "count"),
    )
    
    # Add LZ price if available
    if lz_col and lz_col in df.columns:
        agg["avg_price"] = tmp.groupby("year_month")[lz_col].mean()
    else:
        agg["avg_price"] = float("nan")
    
    # Calculate capture rate
    agg["capture_rate"] = (agg["favorable_hrs"] / agg["total_hrs"] * 100).round(1)
    agg.index = agg.index.astype(str)
    agg.index.name = "Month"
    result = agg.reset_index()
    
    # Rename columns for display
    result.columns = [
        "Month", "Avg Spread ($/MWh)", "Max Spread ($/MWh)", "Min Spread ($/MWh)",
        "Volatility (Std)", "Favorable Hrs", "Total Hrs",
        "Avg LZ Price ($/MWh)", "Capture Rate (%)",
    ]
    
    # Round numeric columns
    for col in result.columns:
        if col != "Month" and result[col].dtype in ("float64", "float32"):
            result[col] = result[col].round(2)
    
    return result


# ── Anomaly Detection ──────────────────────────────────────────────────────

# Zone mappings for weather and grid fundamentals
ZONE_TEMP_COL = {
    "coast": "temperature_C",
    "south": "temperature_S",
    "west": "temperature_W",
    "north": "temperature_N",
    "east": "temperature_E",
}

ZONE_HUMIDITY_COL = {
    "coast": "relative_humidity_C",
    "south": "relative_humidity_S",
    "west": "relative_humidity_W",
    "north": "relative_humidity_N",
    "east": "relative_humidity_E",
}

ZONE_HEAT_INDEX_COL = {
    "coast": "COAST_Heat_Index",
    "south": "SOUTH_Heat_Index",
    "west": "WEST_Heat_Index",
    "north": "NORTH_Heat_Index",
    "east": "EAST_Heat_Index",
}

LZ_PRICE_COLS = {
    "LZ_HOUSTON_DAM": "Houston",
    "LZ_SOUTH_DAM": "South",
    "LZ_WEST_DAM": "West",
    "LZ_NORTH_DAM": "North",
    "LZ_RAYBN_DAM": "Rayburn",
}


def zone_from_column(col: str) -> str | None:
    """Extract zone from spread or LZ column name."""
    if col.startswith("spread_"):
        parts = col.replace("spread_", "").split("_")
        if parts:
            zone_map = {"h": "coast", "s": "south", "w": "west", "n": "north", "r": "east"}
            return zone_map.get(parts[0])
    elif col in LZ_PRICE_COLS:
        lz_to_zone = {
            "LZ_HOUSTON_DAM": "coast",
            "LZ_SOUTH_DAM": "south",
            "LZ_WEST_DAM": "west",
            "LZ_NORTH_DAM": "north",
            "LZ_RAYBN_DAM": "east",
        }
        return lz_to_zone.get(col)
    return None


def detect_anomalous_days(
    df: pd.DataFrame,
    spread_col: str,
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Identify days where the daily average spread is anomalous (|z-score| > threshold).
    
    Returns DataFrame indexed by date with columns:
        avg_spread, z_score, direction ('Spike' or 'Crash')
    sorted by |z_score| descending.
    """
    if df.empty or spread_col not in df.columns:
        return pd.DataFrame()
    
    # Resample to daily averages
    daily = df[spread_col].resample("D").mean().dropna()
    if len(daily) < 10:  # Need sufficient data for meaningful statistics
        return pd.DataFrame()
    
    # Calculate z-scores
    mean = daily.mean()
    std = daily.std()
    if std == 0:
        return pd.DataFrame()
    
    z = (daily - mean) / std
    mask = z.abs() >= z_threshold
    
    anomalies = pd.DataFrame({
        "avg_spread": daily[mask].round(2),
        "z_score": z[mask].round(2),
    })
    anomalies["direction"] = anomalies["z_score"].apply(
        lambda x: "Spike" if x > 0 else "Crash"
    )
    
    # Sort by absolute z-score (most extreme first)
    anomalies = anomalies.reindex(
        anomalies["z_score"].abs().sort_values(ascending=False).index
    )
    
    return anomalies


def anomaly_scatter_chart(
    df: pd.DataFrame,
    spread_col: str,
    anomalies: pd.DataFrame,
    col_label: str = "Spread",
    spread_label: str | None = None,
) -> go.Figure:
    """Spread time series with anomalous days highlighted as large markers."""
    if df.empty or spread_col not in df.columns:
        return go.Figure()
    
    # Resample to daily averages
    daily = df[spread_col].resample("D").mean().dropna()
    if daily.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Normal days as line
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily.values,
        mode="lines", name="Daily Avg Spread",
        line=dict(color="#94a3b8", width=1),
        opacity=0.5,
        hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}/MWh<extra></extra>",
    ))
    
    if not anomalies.empty:
        # Spikes (positive anomalies) - use blue for terminal theme
        spikes = anomalies[anomalies["direction"] == "Spike"]
        if not spikes.empty:
            fig.add_trace(go.Scatter(
                x=spikes.index, y=spikes["avg_spread"],
                mode="markers", name="Spike",
                marker=dict(
                    size=10, 
                    color="#58a6ff",  # Blue for spikes (terminal theme)
                    symbol="triangle-up", 
                    line=dict(width=1, color="#0a0e17")
                ),
                hovertemplate="SPIKE<br>%{x|%b %d %Y}<br>$%{y:.2f}/MWh<br>z=%{customdata:.1f}<extra></extra>",
                customdata=spikes["z_score"],
            ))
        
        # Crashes (negative anomalies) - use lighter blue/grey
        crashes = anomalies[anomalies["direction"] == "Crash"]
        if not crashes.empty:
            fig.add_trace(go.Scatter(
                x=crashes.index, y=crashes["avg_spread"],
                mode="markers", name="Crash",
                marker=dict(
                    size=10, 
                    color="#7ab8ff",  # Lighter blue for crashes
                    symbol="triangle-down", 
                    line=dict(width=1, color="#0a0e17")
                ),
                hovertemplate="CRASH<br>%{x|%b %d %Y}<br>$%{y:.2f}/MWh<br>z=%{customdata:.1f}<extra></extra>",
                customdata=crashes["z_score"],
            ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
    
    # Apply dark theme
    title_suffix = f" — {spread_label}" if spread_label else ""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        height=420,
        title=f"Anomalous {col_label} Days{title_suffix}",
        yaxis_title=f"Avg {col_label} ($/MWh)",
        xaxis=dict(gridcolor="#1e2d40"),
        yaxis=dict(gridcolor="#1e2d40"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    
    return fig


def build_anomaly_fundamentals(
    df: pd.DataFrame,
    anomaly_dates: pd.DatetimeIndex,
    region: str,
    anom_col: str | None = None,
) -> pd.DataFrame:
    """
    For each anomalous date, compute daily averages of all fundamentals.
    
    Returns DataFrame with one row per anomalous day.
    """
    if df.empty or len(anomaly_dates) == 0:
        return pd.DataFrame()
    
    # Determine effective region for column mapping
    effective_region = region
    if region == "all" and anom_col:
        effective_region = zone_from_column(anom_col) or "coast"
    
    # Map fundamental labels to column names
    col_mapping = {
        "Temp (°C)": ZONE_TEMP_COL.get(effective_region),
        "Humidity (%)": ZONE_HUMIDITY_COL.get(effective_region),
        "Heat Index (°F)": ZONE_HEAT_INDEX_COL.get(effective_region),
        "Wind Gen (MW)": ZONE_WIND_COL.get(effective_region),
        "Solar Gen (MW)": ZONE_SOLAR_COL.get(effective_region),
        "Load (MW)": ZONE_LOAD_MAP.get(effective_region),
        "Net Load (MW)": ZONE_NET_LOAD_MAP.get(effective_region),
    }
    
    # Filter to only available columns
    fundamental_cols = {label: col for label, col in col_mapping.items() 
                       if col and col in df.columns}
    
    if not fundamental_cols:
        return pd.DataFrame()
    
    # Build rows for each anomalous day
    rows = []
    for dt in anomaly_dates:
        day_mask = df.index.date == dt.date() if hasattr(dt, 'date') else df.index.date == dt
        day_df = df[day_mask]
        if day_df.empty:
            continue
        
        row = {"Date": dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)}
        for label, col in fundamental_cols.items():
            row[label] = round(day_df[col].mean(), 1) if col in day_df.columns else None
        rows.append(row)
    
    return pd.DataFrame(rows)


def anomaly_fundamentals_radar(
    anomaly_row: dict,
    baseline_row: dict,
    date_label: str,
) -> go.Figure:
    """Radar chart comparing anomalous day fundamentals vs baseline averages."""
    categories = list(anomaly_row.keys())
    if not categories:
        return go.Figure()
    
    anom_vals = [anomaly_row[k] for k in categories]
    base_vals = [baseline_row[k] for k in categories]
    
    # Normalize to baseline = 100%
    norm_anom = []
    norm_base = []
    for a, b in zip(anom_vals, base_vals):
        if b and b != 0:
            norm_anom.append(round(a / b * 100, 1) if a else 0)
            norm_base.append(100.0)
        else:
            norm_anom.append(0)
            norm_base.append(100.0)
    
    fig = go.Figure()
    
    # Baseline (dashed circle at 100%)
    fig.add_trace(go.Scatterpolar(
        r=norm_base + [norm_base[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(148,163,184,0.1)",
        line=dict(color="#94a3b8", width=1, dash="dash"),
        name="Baseline (100%)",
    ))
    
    # Anomalous day
    fig.add_trace(go.Scatterpolar(
        r=norm_anom + [norm_anom[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(88,166,255,0.15)",
        line=dict(color="#58a6ff", width=2),
        name=date_label,
    ))
    
    max_val = max(norm_anom + [150]) if norm_anom else 150
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        height=350,
        title=f"Fundamentals — {date_label}",
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(
                visible=True, 
                gridcolor="#1e2d40",
                ticksuffix="%", 
                range=[0, max_val]
            ),
            angularaxis=dict(gridcolor="#1e2d40"),
        ),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        showlegend=True,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    
    return fig


def cumulative_revenue_with_drawdown(drawdown_df: pd.DataFrame) -> go.Figure:
    """
    Two-panel chart: cumulative revenue (top) and drawdown (bottom).
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
    )

    # Top: cumulative revenue and running max
    fig.add_trace(
        go.Scatter(
            x=drawdown_df["date"],
            y=drawdown_df["cumulative_revenue"],
            name="Cumulative Revenue",
            line=dict(color=CHART_THEME["accent"], width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.10)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=drawdown_df["date"],
            y=drawdown_df["running_max"],
            name="Peak",
            line=dict(color="#94a3b8", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Bottom: drawdown bars (negative)
    fig.add_trace(
        go.Bar(
            x=drawdown_df["date"],
            y=drawdown_df["drawdown"],
            name="Drawdown",
            marker_color="#1e40af",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    _apply_base_layout(fig, "Cumulative Revenue & Drawdown", height=500)
    fig.update_yaxes(
        title_text="Cumulative Revenue ($)",
        row=1,
        col=1,
        gridcolor=CHART_THEME["gridcolor"],
    )
    fig.update_yaxes(
        title_text="Drawdown ($)",
        row=2,
        col=1,
        gridcolor=CHART_THEME["gridcolor"],
    )
    fig.update_layout(
        legend=dict(orientation="h", y=1.06, x=0),
    )
    return fig


def monthly_revenue_box_plot(analysis_df: pd.DataFrame) -> go.Figure:
    """
    Box plot of daily revenue grouped by calendar month.
    """
    df = analysis_df.copy()
    df["month_label"] = pd.to_datetime(df["date"]).dt.strftime("%b")
    df["month_num"] = pd.to_datetime(df["date"]).dt.month

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    present_months = sorted(df["month_num"].unique())
    ordered_labels = [month_order[m - 1] for m in present_months]

    fig = go.Figure()
    for m_num, m_label in zip(present_months, ordered_labels):
        month_data = df[df["month_num"] == m_num]["dam_revenue"]
        if month_data.empty:
            continue
        color = CHART_THEME["accent"]
        fig.add_trace(
            go.Box(
                y=month_data,
                name=m_label,
                marker_color=color,
                line_color=color,
                boxmean=True,
            )
        )

    fig.add_hline(
        y=0,
        line_color=CHART_THEME["zero_line_color"],
        line_width=1,
    )

    _apply_base_layout(
        fig,
        "Daily Revenue by Month — Seasonal Reliability",
        height=STANDARD_HEIGHT,
    )
    fig.update_layout(
        yaxis_title="Daily Revenue ($)",
        showlegend=False,
    )
    return fig


# ── Similar Days Lookup ──────────────────────────────────────────────────

def find_similar_days(
    df: pd.DataFrame,
    target_values: dict,
    region: str,
    spread_col: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    Find the N most similar historical days to a set of target fundamental values.

    Uses normalized Euclidean distance across available fundamentals.

    Parameters
    ----------
    df            : full (unfiltered) regional DataFrame
    target_values : dict mapping feature labels → target values
    region        : region name
    spread_col    : spread column for computing daily spread
    n             : number of similar days to return

    Returns
    -------
    DataFrame with columns: Date, distance, avg_spread, + each fundamental
    """
    if df.empty or spread_col not in df.columns:
        return pd.DataFrame()

    # When region="all", infer zone from the spread column
    effective_region = region
    if region == "all":
        effective_region = zone_from_column(spread_col) or "coast"

    # Map user-facing labels to data columns
    label_to_col = {}
    for label, col in [
        ("Temp (°C)", ZONE_TEMP_COL.get(effective_region)),
        ("Humidity (%)", ZONE_HUMIDITY_COL.get(effective_region)),
        ("Heat Index (°F)", ZONE_HEAT_INDEX_COL.get(effective_region)),
        ("Wind Gen (MW)", ZONE_WIND_COL.get(effective_region)),
        ("Solar Gen (MW)", ZONE_SOLAR_COL.get(effective_region)),
        ("Load (MW)", ZONE_LOAD_MAP.get(effective_region)),
    ]:
        if col and col in df.columns and label in target_values:
            label_to_col[label] = col

    if not label_to_col:
        return pd.DataFrame()

    # Build daily averages
    daily_cols = list(label_to_col.values()) + [spread_col]
    daily = df[daily_cols].resample("D").mean().dropna()
    if len(daily) < 5:
        return pd.DataFrame()

    # Normalize using z-scores
    means = daily.mean()
    stds = daily.std().replace(0, 1)
    normalized = (daily - means) / stds

    # Compute distance from target
    target_norm = {}
    for label, col in label_to_col.items():
        val = target_values[label]
        target_norm[col] = (val - means[col]) / stds[col]

    dist = pd.Series(0.0, index=normalized.index)
    for col, norm_val in target_norm.items():
        dist += (normalized[col] - norm_val) ** 2
    dist = dist ** 0.5

    # Get top N
    top_idx = dist.nsmallest(n).index
    result = daily.loc[top_idx].copy()
    result["Distance"] = dist.loc[top_idx].round(3)
    outcome_label = "Avg Price ($/MWh)" if "spread_" not in spread_col else "Avg Spread ($/MWh)"
    result[outcome_label] = result[spread_col].round(2)

    # Rename columns to friendly labels
    for label, col in label_to_col.items():
        result[label] = result[col].round(1)

    # Clean up
    keep_cols = ["Distance", outcome_label] + list(label_to_col.keys())
    result = result[keep_cols]
    result.index.name = "Date"
    result = result.reset_index()
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    result = result.sort_values("Distance")

    return result


def similar_days_spread_distribution(
    similar_days_df: pd.DataFrame,
    col_label: str = "Spread",
) -> go.Figure:
    """Histogram of spreads/prices from similar days with stats annotation."""
    outcome_col = "Avg Spread ($/MWh)" if "Avg Spread ($/MWh)" in similar_days_df.columns else "Avg Price ($/MWh)"
    if similar_days_df.empty or outcome_col not in similar_days_df.columns:
        return go.Figure()

    values = similar_days_df[outcome_col]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=max(5, len(values) // 2),
        marker_color="#58a6ff",  # Blue accent for terminal theme
        opacity=0.8,
        name="Similar Days",
        hovertemplate=f"{col_label}: $%{{x:.2f}}/MWh<br>Count: %{{y}}<extra></extra>",
    ))

    avg = values.mean()
    fig.add_vline(
        x=avg,
        line_dash="dash",
        line_color="#7ab8ff",  # Lighter blue for terminal theme
        line_width=2,
        annotation_text=f"Mean: ${avg:.2f}",
        annotation_font=dict(color="#7ab8ff", size=11)
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        height=350,
        title=f"{col_label} Distribution of Similar Days",
        xaxis_title=f"Avg {col_label} ($/MWh)",
        yaxis_title="Count",
        xaxis=dict(gridcolor="#1e2d40"),
        yaxis=dict(gridcolor="#1e2d40"),
        showlegend=False,
        margin=dict(l=60, r=20, t=80, b=40),
    )

    return fig


# ── Regime Signal Calculation ──────────────────────────────────────────────

def compute_regime_signal(
    df: pd.DataFrame,
    spread_col: str,
) -> dict:
    """
    Compute a trading regime signal based on spread momentum and volatility.

    Parameters
    ----------
    df : DataFrame with datetime index
    spread_col : Column name containing spread or price values (e.g., "spread_h_s" or "LZ_HOUSTON_DAM")

    Returns
    -------
    dict with keys:
        regime: 'Bullish' / 'Bearish' / 'Neutral' / 'N/A'
        momentum_7d: 7-day spread momentum (current 7d avg vs prior 7d avg)
        momentum_30d: 30-day trend direction
        vol_regime: 'High Vol' / 'Low Vol' / 'Normal' / 'N/A'
        signal_strength: 0-100 composite score
    """
    if df.empty or spread_col not in df.columns:
        return {
            "regime": "N/A",
            "momentum_7d": 0,
            "momentum_30d": 0,
            "vol_regime": "N/A",
            "signal_strength": 0,
        }

    # Resample to daily averages (required for regime calculation)
    daily = df[spread_col].resample("D").mean().dropna()
    
    # Need at least 30 days of data
    if len(daily) < 30:
        return {
            "regime": "N/A",
            "momentum_7d": 0,
            "momentum_30d": 0,
            "vol_regime": "N/A",
            "signal_strength": 0,
        }

    # ── 7-Day Momentum Calculation ─────────────────────────────────────────────
    # Compare last 7 days average vs prior 7 days average
    last_7d = daily.iloc[-7:].mean()
    prior_7d = daily.iloc[-14:-7].mean() if len(daily) >= 14 else daily.mean()
    momentum_7d = last_7d - prior_7d
    # Positive = recent trend is up, Negative = recent trend is down

    # ── 30-Day Trend Calculation ──────────────────────────────────────────────
    # Calculate slope of 30-day rolling mean (medium-term trend)
    rolling_30 = daily.rolling(30).mean().dropna()
    if len(rolling_30) >= 7:
        momentum_30d = rolling_30.iloc[-1] - rolling_30.iloc[-7]
    else:
        momentum_30d = 0
    # Positive = medium-term trend is up, Negative = medium-term trend is down

    # ── Volatility Regime Calculation ────────────────────────────────────────
    # Compare recent volatility (14D) to long-term volatility (60D)
    vol_14d = daily.iloc[-14:].std() if len(daily) >= 14 else daily.std()
    vol_60d = daily.iloc[-60:].std() if len(daily) >= 60 else daily.std()
    vol_ratio = vol_14d / vol_60d if vol_60d > 0 else 1.0

    if vol_ratio > 1.3:
        vol_regime = "High Vol"  # Recent volatility is 30%+ higher than long-term
    elif vol_ratio < 0.7:
        vol_regime = "Low Vol"   # Recent volatility is 30%+ lower than long-term
    else:
        vol_regime = "Normal"     # Volatility is within normal range

    # ── Regime Classification ──────────────────────────────────────────────────
    # Bullish: Both short-term (7D) and medium-term (30D) momentum are positive
    # Bearish: Both short-term (7D) and medium-term (30D) momentum are negative
    # Neutral: Mixed signals (one positive, one negative)
    if momentum_7d > 0 and momentum_30d > 0:
        regime = "Bullish"
    elif momentum_7d < 0 and momentum_30d < 0:
        regime = "Bearish"
    else:
        regime = "Neutral"

    # ── Signal Strength Calculation ──────────────────────────────────────────
    # Strength is based on magnitude of momentum relative to historical volatility
    # Higher strength = stronger signal relative to normal volatility
    avg_vol = daily.std()
    if avg_vol > 0:
        # Weighted: 7D momentum (50%) + 30D momentum (30%)
        strength = min(100, int(abs(momentum_7d) / avg_vol * 50 + abs(momentum_30d) / avg_vol * 30))
    else:
        strength = 0

    return {
        "regime": regime,
        "momentum_7d": round(momentum_7d, 2),
        "momentum_30d": round(momentum_30d, 2),
        "vol_regime": vol_regime,
        "signal_strength": strength,
    }

