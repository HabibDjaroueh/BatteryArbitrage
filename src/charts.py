from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REGION_HEAT_INDEX = {
    "coast": "COAST_Heat_Index",
    "south": "SOUTH_Heat_Index",
    "west": "WEST_Heat_Index",
    "north": "NORTH_Heat_Index",
    "east": "EAST_Heat_Index",
}

SPREAD_COLOR_MAP = {
    "spread_h_s": "#38bdf8",   # sky blue    — Houston vs South
    "spread_h_n": "#34d399",   # emerald     — Houston vs North
    "spread_h_w": "#fb923c",   # orange      — Houston vs West
    "spread_h_r": "#f472b6",   # pink        — Houston vs East
    "spread_n_s": "#a78bfa",   # purple      — North vs South
    "spread_n_w": "#fbbf24",   # amber       — North vs West
    "spread_n_r": "#e879f9",   # fuchsia     — North vs East
    "spread_s_w": "#4ade80",   # green       — South vs West
    "spread_s_r": "#f87171",   # red         — South vs East
    "spread_w_r": "#67e8f9",   # cyan        — West vs East
    "spread_r_n": "#818cf8",   # indigo      — East vs North
    "spread_r_w": "#fb7185",   # rose        — East vs West
}


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

    for spread_col in spread_cols[:2]:   # hard cap at 2
        if spread_col not in df.columns:
            continue

        color   = SPREAD_COLOR_MAP.get(spread_col, "#94a3b8")
        label   = _spread_label(spread_col)
        daily   = df[spread_col].resample("D").mean()
        rolling = daily.rolling(30).mean()

        # Daily spread line
        fig.add_trace(go.Scatter(
            x=daily.index,
            y=daily.values,
            name=label,
            mode="lines",
            line=dict(color=color, width=1.5),
            opacity=0.85,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "%{x|%b %d %Y}<br>"
                "Spread: $%{y:.2f}/MWh<extra></extra>"
            )
        ))

        # 30D rolling mean — dotted, same color
        if show_rolling:
            fig.add_trace(go.Scatter(
                x=rolling.index,
                y=rolling.values,
                name=f"{label} 30D MA",
                mode="lines",
                line=dict(color=color, width=2, dash="dot"),
                opacity=0.5,
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

        # Histogram trace
        fig.add_trace(go.Histogram(
            x=values,
            name=label,
            nbinsx=80,
            marker_color=color,
            opacity=0.55 if len(spread_cols) == 2 else 0.75,
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
            line=dict(color=color, width=1.5, dash="dash")
        )
        fig.add_annotation(
            x=p50,
            y=0.92 - (i * 0.15),   # second spread annotation sits lower
            yref="paper",
            text=f"<b>{label} P50: ${p50:.1f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            arrowwidth=1,
            ax=45, ay=0,
            font=dict(color=color, size=10),
            bgcolor="#0a0e17",
            bordercolor=color,
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
                    line=dict(color=color, width=1, dash="dot")
                )
                fig.add_annotation(
                    x=val,
                    y=y_anchor,
                    yref="paper",
                    text=f"<b>{pct_label}</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    arrowwidth=1,
                    ax=45, ay=0,
                    font=dict(color=color, size=10),
                    bgcolor="#0a0e17",
                    bordercolor=color,
                    borderwidth=1,
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
        (wind_col, "#a78bfa", "Wind"),
        (solar_col, "#fbbf24", "Solar"),
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

        marker_colors = [
            "#34d399" if v >= 0 else "#f87171"
            for v in binned["avg_spread"]
        ]

        fig.add_trace(
            go.Scatter(
                x=binned["avg_re"],
                y=binned["avg_spread"],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=8, color=marker_colors),
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

