"""DEPRECATED — not used by any Streamlit page.

As of the forecast-driven DAM upgrade, the Battery Simulation page
(pages/4_Battery_Simulation.py) no longer uses revenue_time_series(),
revenue_histogram(), monthly_revenue_chart(), or sensitivity_heatmap().
No other page in pages/ imports this module. These chart helpers are
retained for reference and for any existing tests or scripts that
still call them.

The Battery Simulation page now builds its charts inline with Plotly
(forecast vs actual scatter, daily revenue PF vs DAM, daily error cost,
etc.) and uses data from src/forecast_error (e.g. monthly_error_breakdown).

Do not add new chart logic here; add it to the page or to a dedicated
forecast-error chart module if needed.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def revenue_time_series(sim_df: pd.DataFrame) -> go.Figure:
    """
    Daily net revenue time series with 30-day rolling mean overlay
    and cumulative revenue on secondary y-axis.
    """
    sim_df   = sim_df.copy()
    sim_df["date"]       = pd.to_datetime(sim_df["date"])
    sim_df["rolling_30"] = sim_df["net_revenue"].rolling(30).mean()
    sim_df["cumulative"] = sim_df["net_revenue"].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Daily revenue bars
    colors = [
        "#34d399" if v > 0 else "#f87171"
        for v in sim_df["net_revenue"]
    ]
    fig.add_trace(go.Bar(
        x=sim_df["date"],
        y=sim_df["net_revenue"],
        name="Daily Net Revenue",
        marker_color=colors,
        opacity=0.7,
        hovertemplate=(
            "%{x|%b %d %Y}<br>"
            "Net Revenue: $%{y:,.2f}<extra></extra>"
        )
    ), secondary_y=False)

    # 30D rolling mean
    fig.add_trace(go.Scatter(
        x=sim_df["date"],
        y=sim_df["rolling_30"],
        name="30D Rolling Mean",
        line=dict(color="#fb923c", width=2, dash="dot"),
        hovertemplate=(
            "%{x|%b %d %Y}<br>"
            "30D Mean: $%{y:,.2f}<extra></extra>"
        )
    ), secondary_y=False)

    # Cumulative revenue on secondary axis
    fig.add_trace(go.Scatter(
        x=sim_df["date"],
        y=sim_df["cumulative"],
        name="Cumulative Revenue",
        line=dict(color="#a78bfa", width=2),
        hovertemplate=(
            "%{x|%b %d %Y}<br>"
            "Cumulative: $%{y:,.0f}<extra></extra>"
        )
    ), secondary_y=True)

    fig.add_hline(y=0, line_dash="dash", line_color="#475569",
                  line_width=1, secondary_y=False)

    fig.update_layout(
        title="Daily Battery Arbitrage Revenue",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=60, t=60, b=40),
        height=420,
    )
    fig.update_xaxes(gridcolor="#1e2d40")
    fig.update_yaxes(title_text="Daily Revenue ($)",    gridcolor="#1e2d40", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Revenue ($)", showgrid=False,   secondary_y=True)

    return fig


def revenue_histogram(sim_df: pd.DataFrame) -> go.Figure:
    """
    Distribution of daily net revenue with P10/P50/P90 markers.
    Separates dispatched days from non-dispatched (zero revenue) days.
    """
    dispatched     = sim_df[sim_df["dispatched"]]["net_revenue"]
    non_dispatched = sim_df[~sim_df["dispatched"]]["net_revenue"]

    p10 = dispatched.quantile(0.10)
    p50 = dispatched.quantile(0.50)
    p90 = dispatched.quantile(0.90)

    fig = go.Figure()

    # Non-dispatched days
    if len(non_dispatched) > 0:
        fig.add_trace(go.Histogram(
            x=non_dispatched,
            name=f"Not Dispatched ({len(non_dispatched)} days)",
            marker_color="#475569",
            opacity=0.6,
            nbinsx=20,
            hovertemplate="Revenue: $%{x:,.2f}<br>Count: %{y}<extra></extra>"
        ))

    # Dispatched days
    fig.add_trace(go.Histogram(
        x=dispatched,
        name=f"Dispatched ({len(dispatched)} days)",
        marker_color="#38bdf8",
        opacity=0.7,
        nbinsx=60,
        hovertemplate="Revenue: $%{x:,.2f}<br>Count: %{y}<extra></extra>"
    ))

    # P10 / P50 / P90 lines
    for val, color, label in [
        (p10, "#f472b6", f"P10: ${p10:,.0f}"),
        (p50, "#fb923c", f"P50: ${p50:,.0f}"),
        (p90, "#34d399", f"P90: ${p90:,.0f}"),
    ]:
        fig.add_shape(
            type="line", x0=val, x1=val, y0=0, y1=1, yref="paper",
            line=dict(color=color, width=1.5, dash="dash")
        )
        fig.add_annotation(
            x=val, y=0.85, yref="paper",
            text=f"<b>{label}</b>",
            showarrow=True, arrowhead=2, arrowcolor=color,
            arrowwidth=1, ax=40, ay=0,
            font=dict(color=color, size=11),
            bgcolor="#0a0e17", bordercolor=color,
            borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        title="Daily Revenue Distribution — Dispatched Days",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        barmode="overlay",
        xaxis=dict(gridcolor="#1e2d40", title="Daily Net Revenue ($)"),
        yaxis=dict(gridcolor="#1e2d40", title="Count"),
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=20, t=60, b=40),
        height=380,
    )

    return fig


def monthly_revenue_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """
    Monthly total net revenue bar chart with avg daily revenue line overlay.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = [
        "#34d399" if v >= 0 else "#f87171"
        for v in monthly_df["Total Net Revenue ($)"]
    ]

    fig.add_trace(go.Bar(
        x=monthly_df["Month"],
        y=monthly_df["Total Net Revenue ($)"],
        name="Monthly Revenue",
        marker_color=colors,
        opacity=0.8,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Total Revenue: $%{y:,.0f}<extra></extra>"
        )
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly_df["Month"],
        y=monthly_df["Avg Daily Revenue ($)"],
        name="Avg Daily Revenue",
        mode="lines+markers",
        line=dict(color="#fb923c", width=2),
        marker=dict(size=6),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Avg Daily: $%{y:,.2f}<extra></extra>"
        )
    ), secondary_y=True)

    fig.add_hline(y=0, line_dash="dash", line_color="#475569",
                  line_width=1, secondary_y=False)

    fig.update_layout(
        title="Monthly Battery Revenue",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=60, t=60, b=60),
        height=400,
        xaxis=dict(gridcolor="#1e2d40", tickangle=-45),
    )
    fig.update_yaxes(title_text="Monthly Revenue ($)",  gridcolor="#1e2d40", secondary_y=False)
    fig.update_yaxes(title_text="Avg Daily Revenue ($)", showgrid=False,    secondary_y=True)

    return fig


def sensitivity_heatmap(sensitivity_df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Renders a heatmap of a chosen metric across efficiency × duration grid.

    Parameters
    ----------
    sensitivity_df : output of run_sensitivity()
    metric         : column name to plot e.g. "Total Revenue ($)"
    """
    pivot = sensitivity_df.pivot(
        index="Efficiency (%)",
        columns="Duration (h)",
        values=metric
    )

    if "Dispatch" in metric or "Rate" in metric:
        text = [[f"{v:.1f}%" for v in row] for row in pivot.values]
    else:
        text = [[f"${v:,.0f}" for v in row] for row in pivot.values]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{d}h" for d in pivot.columns],
        y=[f"{e}%" for e in pivot.index],
        colorscale="Blues",
        text=text,
        texttemplate="%{text}",
        hovertemplate=(
            "Efficiency: %{y}<br>"
            "Duration: %{x}<br>"
            f"{metric}: %{{text}}<extra></extra>"
        ),
        colorbar=dict(title=metric, thickness=14),
    ))

    fig.update_layout(
        title=f"Sensitivity — {metric} across Efficiency × Duration",
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(family="monospace", color="#94a3b8", size=11),
        xaxis=dict(title="Duration (hours)"),
        yaxis=dict(title="Round-Trip Efficiency"),
        margin=dict(l=80, r=40, t=60, b=40),
        height=380,
    )

    return fig
