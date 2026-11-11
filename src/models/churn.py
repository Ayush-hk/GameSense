"""Churn model training and evaluation.

Protocol
--------
1. Candidates: prior-only Dummy baseline, Logistic Regression, Random Forest (class-weighted).
2. Each candidate is fitted on ``fit`` (train avatars, older cutoffs) and scored on ``valid``
   (held-out avatars, newest training cutoff). The best non-dummy model by validation ROC-AUC
   (ties -> PR-AUC) is *selected*, and its decision threshold is tuned for F1 on validation.
3. All candidates are then refitted on ``refit`` (train+valid avatars, all training cutoffs)
   and evaluated ONCE on ``test`` (held-out avatars at the newest cutoff). Test data never
   influences model or threshold selection.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.errors import EmptyDatasetError
from src.features.churn_target import ChurnDataset

MODEL_NAMES = ("dummy_prior", "recency_only", "logistic_regression", "random_forest")
BASELINES = ("dummy_prior", "recency_only")      # reference points, never selected as "the model"


def build_pipeline(name: str, numeric: list[str], categorical: list[str], cfg: dict[str, Any]) -> Pipeline:
    seed = int(cfg["churn"]["random_state"])
    if name == "recency_only":                       # honest heuristic baseline: one feature, one coefficient
        numeric, categorical = ["days_since_last_active"], []
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if name in ("logistic_regression", "recency_only"):
        num_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            *([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)] if categorical else []),
        ],
        verbose_feature_names_out=False,
    )
    mc = cfg["models"]
    if name == "dummy_prior":
        clf: Any = DummyClassifier(strategy="prior")
    elif name == "recency_only":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    elif name == "logistic_regression":
        clf = LogisticRegression(
            C=float(mc["logistic_regression"]["C"]),
            max_iter=int(mc["logistic_regression"]["max_iter"]),
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(mc["random_forest"]["n_estimators"]),
            min_samples_leaf=int(mc["random_forest"]["min_samples_leaf"]),
            max_depth=mc["random_forest"]["max_depth"],
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("prep", prep), ("clf", clf)])


def binary_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else None,
        "accuracy": float((tn + tp) / max(len(y), 1)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def bootstrap_ci(y, scores: dict[str, np.ndarray], groups, reference: str, n_boot: int = 500, seed: int = 0) -> dict[str, Any]:
    """Cluster bootstrap over AVATARS (an avatar can appear in several test periods).

    Returns 95% percentile intervals of ROC-AUC / PR-AUC per model and of the AUC difference vs ``reference``.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.RandomState(seed)
    acc: dict[str, dict[str, list[float]]] = {n: {"roc_auc": [], "pr_auc": []} for n in scores}
    diff: dict[str, list[float]] = {n: [] for n in scores if n != reference}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_group[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs = {}
        for n, s in scores.items():
            aucs[n] = roc_auc_score(y[idx], s[idx])
            acc[n]["roc_auc"].append(aucs[n])
            acc[n]["pr_auc"].append(average_precision_score(y[idx], s[idx]))
        for n in diff:
            diff[n].append(aucs[n] - aucs[reference])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    return {
        "n_boot": n_boot,
        "n_avatars": int(len(uniq)),
        "reference": reference,
        "models": {n: {k: q(v) for k, v in d.items()} for n, d in acc.items()},
        "roc_auc_difference_vs_reference": {n: q(v) for n, v in diff.items()},
    }


def best_f1_threshold(y_true, proba) -> float:
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def _xy(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    return df[numeric + categorical], df["churned"].to_numpy().astype(int)


def _positive_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipe.predict_proba(X)
    classes = list(pipe.named_steps["clf"].classes_)
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def feature_rules(cfg: dict[str, Any]) -> dict[str, list[float | None]]:
    """Plausible [min, max] per feature, used to validate user-supplied prediction input."""
    obs = float(cfg["churn"]["observation_days"])
    lo, hi = cfg["dataset"]["level_range"]
    return {
        "active_days": [0, obs],
        "est_hours_online": [0, None],
        "n_sessions": [0, None],
        "mean_session_min": [0, None],
        "max_session_min": [0, None],
        "sessions_per_active_day": [0, None],
        "days_since_last_active": [0, obs],
        "active_days_last7": [0, 7],
        "active_days_prev7": [0, 7],
        "level_last": [lo, hi],
        "level_gain": [0, hi - lo],
        "zones_visited": [0, None],
        "has_guild": [0, 1],
        "guild_changes": [0, None],
        "weekend_share": [0, 1],
        "night_share": [0, 1],
        "observed_tenure_days": [0, None],
    }


def train_and_evaluate(ds: ChurnDataset, cfg: dict[str, Any], paths=None, data_origin: str = "user_provided_file") -> dict[str, Any]:
    cols = set(ds.snapshots.columns)
    numeric = [f for f in cfg["churn"]["numeric_features"] if f in cols]       # drop features the data cannot provide
    categorical = [f for f in cfg["churn"]["categorical_features"] if f in cols]
    unavailable = [f for f in cfg["churn"]["numeric_features"] + cfg["churn"]["categorical_features"] if f not in cols]
    for name, d in (("fit", ds.fit), ("valid", ds.valid), ("test", ds.test), ("refit", ds.refit)):
        if len(d) == 0:
            raise EmptyDatasetError(f"The '{name}' churn set is empty; the dataset is too small for this protocol.")
        if d["churned"].nunique() < 2 and name in ("fit", "refit"):
            raise EmptyDatasetError(f"The '{name}' churn set has a single class; cannot train a classifier.")

    Xf, yf = _xy(ds.fit, numeric, categorical)
    Xv, yv = _xy(ds.valid, numeric, categorical)
    Xr, yr = _xy(ds.refit, numeric, categorical)
    Xt, yt = _xy(ds.test, numeric, categorical)

    # ---- 1) validation round: fit on `fit`, score on `valid`
    validation: dict[str, Any] = {}
    valid_proba: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        pipe = build_pipeline(name, numeric, categorical, cfg).fit(Xf, yf)
        valid_proba[name] = _positive_proba(pipe, Xv)
        validation[name] = binary_metrics(yv, valid_proba[name], 0.5)

    candidates = [n for n in MODEL_NAMES if n not in BASELINES]

    def sort_key(n: str) -> tuple[float, float]:
        m = validation[n]
        return (m["roc_auc"] if m["roc_auc"] is not None else -1.0, m["pr_auc"] if m["pr_auc"] is not None else -1.0)

    selected = max(candidates, key=sort_key)
    threshold = best_f1_threshold(yv, valid_proba[selected])

    # ---- 2) final round: refit on train+valid, evaluate once on test
    final_pipes: dict[str, Pipeline] = {}
    test_metrics: dict[str, Any] = {}
    test_proba: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        pipe = build_pipeline(name, numeric, categorical, cfg).fit(Xr, yr)
        final_pipes[name] = pipe
        test_proba[name] = _positive_proba(pipe, Xt)
        test_metrics[name] = binary_metrics(yt, test_proba[name], 0.5)
    test_selected_tuned = binary_metrics(yt, test_proba[selected], threshold)
    ci = bootstrap_ci(yt, test_proba, ds.test["avatar"].to_numpy(), reference="recency_only", seed=int(cfg["churn"]["random_state"]))

    prevalence = {"fit": float(yf.mean()), "valid": float(yv.mean()), "refit": float(yr.mean()), "test": float(yt.mean())}
    warnings = [
        f"Metrics describe held-out avatars over {ds.spec.n_test_cutoffs} out-of-time period(s) on one game server; "
        "they are not guarantees for other games, servers or periods.",
        "Churn here means 'no activity in the prediction window' for an avatar, which is not the same as a player leaving the game.",
    ]
    if min(prevalence.values()) < 0.05 or max(prevalence.values()) > 0.95:
        warnings.append("Strong class imbalance: prefer PR-AUC, precision and recall over accuracy.")
    lo_d = ci["roc_auc_difference_vs_reference"].get(selected)
    if lo_d and lo_d[0] <= 0:
        warnings.append(
            f"The selected model's ROC-AUC advantage over the recency-only baseline is not statistically clear "
            f"(95% bootstrap interval of the difference {lo_d[0]:+.3f} to {lo_d[1]:+.3f} includes 0)."
        )
    if abs(prevalence["test"] - prevalence["refit"]) > 0.10:
        warnings.append(
            f"Churn prevalence shifts between training ({prevalence['refit']:.1%}) and test ({prevalence['test']:.1%}); "
            "the population is non-stationary over time."
        )

    results = {
        "selected_model": selected,
        "selection_rule": "highest validation ROC-AUC among Logistic Regression and Random Forest (ties: PR-AUC)",
        "threshold_tuned_on_validation": threshold,
        "threshold_rule": "threshold maximising F1 on the validation set",
        "validation": validation,
        "test": test_metrics,
        "test_selected_tuned_threshold": test_selected_tuned,
        "test_bootstrap_ci": ci,
        "sizes": {"fit": len(yf), "valid": len(yv), "refit": len(yr), "test": len(yt)},
        "prevalence": prevalence,
        "features": {"numeric": numeric, "categorical": categorical, "unavailable_in_data": unavailable},
        "warnings": warnings,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_origin": data_origin,
    }

    # ---- curves + report text for the selected model
    fpr, tpr, _ = roc_curve(yt, test_proba[selected]) if len(np.unique(yt)) == 2 else (np.array([0, 1]), np.array([0, 1]), None)
    prec, rec, _ = precision_recall_curve(yt, test_proba[selected]) if len(np.unique(yt)) == 2 else (np.array([1, 0]), np.array([0, 1]), None)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    pr_df = pd.DataFrame({"precision": prec, "recall": rec})
    roc_df = roc_df.iloc[np.linspace(0, len(roc_df) - 1, min(len(roc_df), 600)).astype(int)]   # keep files small
    pr_df = pr_df.iloc[np.linspace(0, len(pr_df) - 1, min(len(pr_df), 600)).astype(int)]
    y_pred_tuned = (test_proba[selected] >= threshold).astype(int)
    report_txt = (
        f"Selected model: {selected}\n\n"
        f"== Test set, default threshold 0.50 ==\n"
        + classification_report(yt, (test_proba[selected] >= 0.5).astype(int), target_names=["retained", "churned"], zero_division=0)
        + f"\n== Test set, threshold {threshold:.3f} (tuned on validation) ==\n"
        + classification_report(yt, y_pred_tuned, target_names=["retained", "churned"], zero_division=0)
    )
    cm = test_selected_tuned["confusion"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["actual_retained", "actual_churned"],
        columns=["predicted_retained", "predicted_churned"],
    )
    comparison = pd.DataFrame(
        [
            {"model": n, "split": split, **{k: m[k] for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy")}}
            for split, block in (("validation", validation), ("test", test_metrics))
            for n, m in block.items()
        ]
    )

    meta = {
        "model_name": selected,
        "threshold": threshold,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_rules": feature_rules(cfg),
        "trained_at": results["trained_at"],
        "data_origin": data_origin,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "churn_definition": ds.definition()["definition"],
        "churn_spec": ds.definition()["spec"],
        "training_rows": int(len(yr)),
        "test_cutoffs": [str(c) for c in ds.plan.test_cutoffs],
        "categories_seen": {c: sorted(ds.snapshots[c].astype(str).unique().tolist()) for c in categorical},
    }

    if paths is not None:
        paths.ensure()
        joblib.dump({"pipeline": final_pipes[selected], "meta": meta}, paths.churn_model)
        paths.churn_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        paths.metrics.write_text(json.dumps(results, indent=2), encoding="utf-8")
        comparison.to_csv(paths.model_comparison, index=False)
        cm_df.to_csv(paths.confusion_matrix)
        paths.classification_report.write_text(report_txt, encoding="utf-8")
        roc_df.to_csv(paths.roc_curve, index=False)
        pr_df.to_csv(paths.pr_curve, index=False)

    return {
        "results": results,
        "pipelines": final_pipes,
        "selected_pipeline": final_pipes[selected],
        "meta": meta,
        "comparison": comparison,
        "confusion_matrix": cm_df,
        "roc": roc_df,
        "pr": pr_df,
        "test_proba": test_proba[selected],
    }
