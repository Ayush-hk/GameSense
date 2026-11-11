"""Plotly figure builders shared by the dashboard and the HTML export in the report stage.

Imports plotly at module level on purpose: the pipeline treats ImportError as "skip figure export".
Every builder returns None when its input file is missing (no placeholder charts).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=50, b=40))


def _read(path):
    return pd.read_csv(path) if path.is_file() else None


def daily_activity_fig(paths):
    df = _read(paths.daily_activity)
    if df is None:
        return None
    fig = px.line(df, x="date", y="active_avatars", title="Distinct avatars seen per day")
    return fig.update_layout(**LAYOUT)


def hourly_heatmap_fig(paths):
    df = _read(paths.hourly_activity)
    if df is None:
        return None
    pivot = df.pivot(index="weekday", columns="hour", values="polls").reindex(range(7)).fillna(0)
    fig = px.imshow(pivot.to_numpy(), x=list(pivot.columns), y=DAYS, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(x="Hour of day", y="Weekday", color="Polls"), title="Activity by weekday and hour (server time as recorded)")
    return fig.update_layout(**LAYOUT)


def cluster_selection_fig(paths):
    df = _read(paths.cluster_selection)
    if df is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["k"], y=df["silhouette"], mode="lines+markers", name="Silhouette"))
    fig.add_trace(go.Scatter(x=df["k"], y=df["davies_bouldin"], mode="lines+markers", name="Davies-Bouldin (lower is better)", yaxis="y2"))
    fig.update_layout(title="Choosing the number of clusters", xaxis_title="k", yaxis_title="Silhouette",
                      yaxis2=dict(title="Davies-Bouldin", overlaying="y", side="right"), **LAYOUT)
    return fig


def roc_fig(paths):
    df = _read(paths.roc_curve)
    if df is None:
        return None
    fig = px.line(df, x="fpr", y="tpr", title="ROC curve (held-out test set)", labels={"fpr": "False positive rate", "tpr": "True positive rate"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    return fig.update_layout(**LAYOUT)


def pr_fig(paths):
    df = _read(paths.pr_curve)
    if df is None:
        return None
    fig = px.line(df, x="recall", y="precision", title="Precision-recall curve (held-out test set)")
    return fig.update_layout(**LAYOUT)


def confusion_fig(paths):
    if not paths.confusion_matrix.is_file():
        return None
    cm = pd.read_csv(paths.confusion_matrix, index_col=0)
    fig = px.imshow(cm.to_numpy(), x=list(cm.columns), y=list(cm.index), text_auto=True, color_continuous_scale="Blues",
                    title="Confusion matrix (test set, tuned threshold)")
    return fig.update_layout(**LAYOUT)


def importance_fig(df: pd.DataFrame, value_col: str, title: str, top: int = 15):
    d = df.head(top).iloc[::-1]
    fig = px.bar(d, x=value_col, y="feature", orientation="h", title=title)
    return fig.update_layout(**LAYOUT)


def export_figures(paths) -> list[str]:
    """Write standalone HTML copies of the available figures into reports/figures."""
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in (("daily_activity", daily_activity_fig), ("hourly_heatmap", hourly_heatmap_fig),
                     ("cluster_selection", cluster_selection_fig), ("roc_curve", roc_fig),
                     ("pr_curve", pr_fig), ("confusion_matrix", confusion_fig)):
        fig = fn(paths)
        if fig is not None:
            out = paths.figures_dir / f"{name}.html"
            fig.write_html(out, include_plotlyjs="cdn")
            written.append(out.name)
    return written
