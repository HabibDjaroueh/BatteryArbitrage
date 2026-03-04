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

