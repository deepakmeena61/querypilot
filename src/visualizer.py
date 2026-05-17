"""
Intelligent chart selection and generation using Plotly.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

PALETTE = ["#1DB954", "#1ed760", "#17a349", "#0d7a36", "#169c45", "#22c55e", "#4ade80", "#86efac"]

CHART_LAYOUT = dict(
    paper_bgcolor="#181818",
    plot_bgcolor="#181818",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#B3B3B3"),
    margin=dict(l=20, r=20, t=60, b=20),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(30,30,30,0.9)",
        bordercolor="#333",
        borderwidth=1,
        font=dict(color="#B3B3B3"),
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#252525",
        linecolor="#333",
        tickfont=dict(size=11, color="#666"),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#252525",
        linecolor="#333",
        tickfont=dict(size=11, color="#666"),
    ),
)


@dataclass
class VisualizationResult:
    figure: Optional[go.Figure]
    chart_type: str
    title: str
    description: str


def _auto_select_chart_type(df: pd.DataFrame, llm_hint: str) -> str:
    """Determine the best chart type for the given DataFrame."""
    if df.empty:
        return "table"

    n_rows, n_cols = df.shape
    if n_rows == 1 and n_cols == 1:
        return "kpi"

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Trust the LLM hint if it maps to something sensible
    valid_hints = {"bar", "horizontal_bar", "line", "pie", "scatter", "heatmap", "kpi", "table"}
    if llm_hint in valid_hints:
        # But override some illogical hints
        if llm_hint == "kpi" and n_rows > 1:
            pass  # fall through to auto
        elif llm_hint == "pie" and n_rows > 8:
            return "bar"
        else:
            return llm_hint

    # Auto selection heuristics
    if len(numeric_cols) == 0:
        return "table"

    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        n_cats = df[cat_col].nunique() if cat_col in df.columns else n_rows
        avg_label_len = df[cat_col].astype(str).str.len().mean() if cat_col in df.columns else 10

        if n_cats <= 6 and n_cols == 2:
            return "pie"
        if avg_label_len > 15 or n_cats > 12:
            return "horizontal_bar"
        return "bar"

    if len(numeric_cols) >= 2 and len(categorical_cols) == 0:
        return "scatter"

    if len(categorical_cols) == 2 and len(numeric_cols) == 1:
        return "heatmap"

    return "table"


def _make_title(question: str, chart_type: str) -> str:
    """Generate a descriptive chart title from the question."""
    q = question.strip().rstrip("?")
    return q[:80] + ("..." if len(q) > 80 else "")


def _format_number(val) -> str:
    if isinstance(val, float):
        return f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _apply_base_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E293B"), x=0, xanchor="left"),
        **CHART_LAYOUT,
    )


def _make_bar(df: pd.DataFrame, cat_col: str, num_col: str, title: str) -> go.Figure:
    df_sorted = df.sort_values(num_col, ascending=False)
    fig = px.bar(
        df_sorted,
        x=cat_col,
        y=num_col,
        color_discrete_sequence=PALETTE,
        text=num_col,
    )
    fig.update_traces(
        texttemplate="%{text:,.1f}",
        textposition="outside",
        marker_color=PALETTE[0],
        marker_line_width=0,
    )
    fig.update_layout(showlegend=False)
    _apply_base_layout(fig, title)
    return fig


def _make_horizontal_bar(df: pd.DataFrame, cat_col: str, num_col: str, title: str) -> go.Figure:
    df_sorted = df.sort_values(num_col, ascending=True).tail(20)
    fig = px.bar(
        df_sorted,
        x=num_col,
        y=cat_col,
        orientation="h",
        color_discrete_sequence=PALETTE,
        text=num_col,
    )
    fig.update_traces(
        texttemplate="%{text:,.2f}",
        textposition="outside",
        marker_color=PALETTE[0],
        marker_line_width=0,
    )
    fig.update_layout(showlegend=False, height=max(400, len(df_sorted) * 28 + 100))
    _apply_base_layout(fig, title)
    return fig


def _make_line(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for i, y_col in enumerate(y_cols[:4]):
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="lines+markers",
                name=y_col,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                marker=dict(size=5),
            )
        )
    _apply_base_layout(fig, title)
    return fig


def _make_pie(df: pd.DataFrame, cat_col: str, num_col: str, title: str) -> go.Figure:
    fig = px.pie(
        df,
        names=cat_col,
        values=num_col,
        color_discrete_sequence=PALETTE,
        hole=0.35,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>",
    )
    _apply_base_layout(fig, title)
    fig.update_layout(showlegend=True)
    return fig


def _make_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: str = None) -> go.Figure:
    kwargs = dict(x=x_col, y=y_col, color_discrete_sequence=PALETTE, opacity=0.6)
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
    fig = px.scatter(df, **kwargs)
    fig.update_traces(marker=dict(size=6))
    _apply_base_layout(fig, title)
    return fig


def _make_heatmap(df: pd.DataFrame, row_col: str, col_col: str, val_col: str, title: str) -> go.Figure:
    pivot = df.pivot_table(index=row_col, columns=col_col, values=val_col, aggfunc="mean")
    fig = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        text_auto=".2f",
        aspect="auto",
    )
    _apply_base_layout(fig, title)
    return fig


def _make_kpi(df: pd.DataFrame, title: str) -> go.Figure:
    val = df.iloc[0, 0]
    col_name = df.columns[0]
    formatted = _format_number(val)

    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=float(val) if isinstance(val, (int, float)) else 0,
            number=dict(
                valueformat=",",
                font=dict(size=72, color=PALETTE[0]),
            ),
            title=dict(text=col_name.replace("_", " ").title(), font=dict(size=20)),
        )
    )
    fig.update_layout(paper_bgcolor="white", height=250, margin=dict(t=40, b=20, l=20, r=20))
    return fig


def create_visualization(
    df: pd.DataFrame,
    question: str,
    llm_hint: str = "table",
) -> VisualizationResult:
    """
    Generate the best visualization for the given DataFrame.
    Returns a VisualizationResult with the Plotly figure and metadata.
    """
    if df.empty:
        return VisualizationResult(
            figure=None,
            chart_type="table",
            title="No data returned",
            description="The query returned no rows.",
        )

    chart_type = _auto_select_chart_type(df, llm_hint)
    title = _make_title(question, chart_type)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    try:
        if chart_type == "kpi":
            fig = _make_kpi(df, title)

        elif chart_type == "bar":
            cat_col = categorical_cols[0] if categorical_cols else df.columns[0]
            num_col = numeric_cols[0] if numeric_cols else df.columns[1]
            fig = _make_bar(df, cat_col, num_col, title)

        elif chart_type == "horizontal_bar":
            cat_col = categorical_cols[0] if categorical_cols else df.columns[0]
            num_col = numeric_cols[0] if numeric_cols else df.columns[1]
            fig = _make_horizontal_bar(df, cat_col, num_col, title)

        elif chart_type == "line":
            x_col = df.columns[0]
            y_cols = numeric_cols if numeric_cols else [df.columns[1]]
            fig = _make_line(df, x_col, y_cols, title)

        elif chart_type == "pie":
            cat_col = categorical_cols[0] if categorical_cols else df.columns[0]
            num_col = numeric_cols[0] if numeric_cols else df.columns[1]
            fig = _make_pie(df, cat_col, num_col, title)

        elif chart_type == "scatter":
            x_col = numeric_cols[0] if len(numeric_cols) >= 1 else df.columns[0]
            y_col = numeric_cols[1] if len(numeric_cols) >= 2 else numeric_cols[0]
            color_col = categorical_cols[0] if categorical_cols else None
            fig = _make_scatter(df, x_col, y_col, title, color_col)

        elif chart_type == "heatmap":
            row_col = categorical_cols[0] if len(categorical_cols) >= 1 else df.columns[0]
            col_col = categorical_cols[1] if len(categorical_cols) >= 2 else df.columns[1]
            val_col = numeric_cols[0] if numeric_cols else df.columns[2]
            fig = _make_heatmap(df, row_col, col_col, val_col, title)

        else:
            # table — no chart
            return VisualizationResult(
                figure=None,
                chart_type="table",
                title=title,
                description="Data is best viewed as a table for this query.",
            )

    except Exception as e:
        logger.warning(f"Chart generation failed ({chart_type}): {e}. Falling back to table.")
        return VisualizationResult(
            figure=None,
            chart_type="table",
            title=title,
            description=f"Chart could not be rendered: {e}",
        )

    return VisualizationResult(
        figure=fig,
        chart_type=chart_type,
        title=title,
        description=f"Displayed as: {chart_type.replace('_', ' ').title()}",
    )
