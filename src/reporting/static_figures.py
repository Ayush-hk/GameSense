"""PNG charts written with matplotlib (always available), so reports have real figures even without Plotly."""
from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import read_json

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def export_png_figures(paths) -> list[str]:
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    f = paths.figures_dir

    if paths.daily_activity.is_file():
        d = pd.read_csv(paths.daily_activity, parse_dates=["date"])
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(d["date"], d["active_avatars"].rolling(7, min_periods=1).mean(), lw=1.4)
        ax.set(title="Distinct avatars seen per day (7-day average)", ylabel="avatars")
        out.append(_save(fig, f / "daily_activity.png"))

    if paths.hourly_activity.is_file():
        h = pd.read_csv(paths.hourly_activity).pivot(index="weekday", columns="hour", values="polls").reindex(range(7)).fillna(0)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        im = ax.imshow(h.to_numpy(), aspect="auto", cmap="Blues")
        ax.set(title="Polls by weekday and hour", xlabel="hour of day", yticks=range(7), yticklabels=DAYS)
        fig.colorbar(im, ax=ax)
        out.append(_save(fig, f / "hourly_heatmap.png"))

    if paths.cluster_selection.is_file():
        c = pd.read_csv(paths.cluster_selection)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(c["k"], c["silhouette"], "o-", label="silhouette")
        ax.set(title="Choosing k", xlabel="k", ylabel="silhouette")
        ax2 = ax.twinx()
        ax2.plot(c["k"], c["davies_bouldin"], "s--", color="tab:orange", label="Davies-Bouldin")
        ax2.set_ylabel("Davies-Bouldin (lower is better)")
        out.append(_save(fig, f / "cluster_selection.png"))

    seg = read_json(paths.segmentation_report)
    if seg:
        z = pd.DataFrame(seg["z_means"]).T
        fig, ax = plt.subplots(figsize=(9, 0.9 + 0.55 * len(z)))
        im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_xticks(range(z.shape[1]), z.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(z)), [f"segment {i}" for i in z.index])
        ax.set_title("Segment means in standard deviations from the overall average")
        fig.colorbar(im, ax=ax)
        out.append(_save(fig, f / "segment_profile_heatmap.png"))

    if paths.roc_curve.is_file() and paths.pr_curve.is_file():
        r, p = pd.read_csv(paths.roc_curve), pd.read_csv(paths.pr_curve)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].plot(r["fpr"], r["tpr"]); axes[0].plot([0, 1], [0, 1], "--", color="grey")
        axes[0].set(title="ROC curve (test)", xlabel="false positive rate", ylabel="true positive rate")
        axes[1].plot(p["recall"], p["precision"])
        axes[1].set(title="Precision-recall (test)", xlabel="recall", ylabel="precision")
        out.append(_save(fig, f / "roc_pr_curves.png"))

    if paths.confusion_matrix.is_file():
        cm = pd.read_csv(paths.confusion_matrix, index_col=0)
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.imshow(cm.to_numpy(), cmap="Blues")
        for (i, j), v in np.ndenumerate(cm.to_numpy()):
            ax.text(j, i, int(v), ha="center", va="center")
        ax.set_xticks([0, 1], ["retained", "churned"]); ax.set_yticks([0, 1], ["retained", "churned"])
        ax.set(title="Confusion matrix (test)", xlabel="predicted", ylabel="actual")
        out.append(_save(fig, f / "confusion_matrix.png"))

    for name, col, title in ((paths.shap_global, "mean_abs_shap", "Mean |SHAP|"),
                             (paths.permutation_importance, "importance_mean", "Permutation importance (drop in test ROC-AUC)")):
        if name.is_file():
            d = pd.read_csv(name).head(12).iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(d["feature"], d[col]); ax.set_title(title)
            out.append(_save(fig, f / f"{name.stem}.png"))

    if paths.churn_by_group.is_file():
        g = pd.read_csv(paths.churn_by_group)
        for dim in ("recency", "active_days_in_window"):
            d = g[g["dimension"] == dim]
            if len(d):
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ax.bar(d["group"], d["churn_rate"] * 100)
                ax.set(title=f"Observed churn rate by {dim.replace('_', ' ')}", ylabel="% churned")
                out.append(_save(fig, f / f"churn_by_{dim}.png"))
    return out
