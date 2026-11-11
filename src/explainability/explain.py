"""Model explanations: SHAP (when installed) with permutation importance as the documented fallback.

Interpretation notes (also shown in the dashboard)
--------------------------------------------------
* SHAP values describe how the *model* uses each feature to move a prediction away from the
  average prediction. They are NOT evidence that changing a feature would change real-world churn.
* Random Forest (TreeExplainer): values are in units of churn *probability*; per row,
  base value + sum(SHAP) = predicted churn probability.
* Logistic Regression (LinearExplainer): values are in *log-odds* of churn.
* One-hot columns (e.g. ``charclass_Mage``) are summed back to their original feature, which is
  exact because SHAP values are additive.
* Permutation importance = drop in ROC-AUC on the held-out test set when one input column is shuffled.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CAUSALITY_WARNING = (
    "Feature attributions explain how the trained model produces its predictions. They are associations "
    "learned from observational data, not proof that changing a behaviour would cause an avatar to stay or leave."
)


def shap_available() -> bool:
    return importlib.util.find_spec("shap") is not None


# ----------------------------------------------------------------------------- helpers
def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    prep = pipeline.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    return pd.DataFrame(np.asarray(prep.transform(X), dtype=float), columns=names, index=X.index)


def _group_columns(names: list[str], categorical: list[str]) -> dict[str, list[str]]:
    """Map each ORIGINAL input feature -> its transformed column names."""
    groups: dict[str, list[str]] = {}
    for name in names:
        owner = next((c for c in categorical if name.startswith(f"{c}_")), name)
        groups.setdefault(owner, []).append(name)
    return groups


def _positive_class_values(sv: Any) -> np.ndarray:
    if isinstance(sv, list):                      # older SHAP: list per class
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                              # newer SHAP: (rows, features, classes)
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    return sv


def _expected_value(explainer: Any) -> float:
    ev = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    return float(ev[-1] if ev.size > 1 else ev[0])


class ShapExplainer:
    """Wraps a SHAP explainer for a fitted churn pipeline (Random Forest or Logistic Regression)."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame, categorical: list[str]):
        if not shap_available():
            raise ImportError("The 'shap' package is not installed.")
        import shap  # local import: optional dependency

        self.pipeline = pipeline
        self.categorical = categorical
        clf = pipeline.named_steps["clf"]
        bg = transform_features(pipeline, background)
        if isinstance(clf, RandomForestClassifier):
            self.explainer = shap.TreeExplainer(clf)
            self.units = "change in predicted churn probability"
            self.kind = "tree"
        elif isinstance(clf, LogisticRegression):
            self.explainer = shap.LinearExplainer(clf, bg)
            self.units = "change in log-odds of churn"
            self.kind = "linear"
        else:
            raise TypeError(f"SHAP explanations are not implemented for {type(clf).__name__}.")
        self.base_value = _expected_value(self.explainer)
        self.names = list(bg.columns)

    def shap_values(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Return (input-level SHAP, transformed-level SHAP, model output being explained)."""
        Xt = transform_features(self.pipeline, X)
        if self.kind == "tree":
            sv = self.explainer.shap_values(Xt.to_numpy(), check_additivity=False)
        else:
            sv = self.explainer.shap_values(Xt.to_numpy())
        sv = _positive_class_values(sv)
        sv_t = pd.DataFrame(sv, columns=self.names, index=X.index)
        groups = _group_columns(self.names, self.categorical)
        sv_in = pd.DataFrame({k: sv_t[v].sum(axis=1) for k, v in groups.items()}, index=X.index)
        clf = self.pipeline.named_steps["clf"]
        if self.kind == "tree":
            output = clf.predict_proba(Xt.to_numpy())[:, list(clf.classes_).index(1)]
        else:
            output = clf.decision_function(Xt.to_numpy())
        return sv_in, sv_t, output

    def additivity_error(self, X: pd.DataFrame) -> float:
        _, sv_t, output = self.shap_values(X)
        return float(np.max(np.abs(self.base_value + sv_t.sum(axis=1).to_numpy() - output)))

    def explain_global(self, X: pd.DataFrame, max_rows: int, seed: int = 42) -> dict[str, Any]:
        if len(X) > max_rows:
            X = X.sample(n=max_rows, random_state=seed)
        sv_in, _, output = self.shap_values(X)
        importance = (
            pd.DataFrame({"feature": sv_in.columns, "mean_abs_shap": sv_in.abs().mean().to_numpy()})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        sample = pd.concat(
            [sv_in.add_prefix("shap__"), X.reset_index(drop=True).set_index(sv_in.index).add_prefix("value__")],
            axis=1,
        )
        return {
            "importance": importance,
            "sample": sample,
            "base_value": self.base_value,
            "units": self.units,
            "additivity_max_abs_error": self.additivity_error(X),
            "n_explained": int(len(X)),
        }

    def explain_local(self, X_row: pd.DataFrame) -> dict[str, Any]:
        if len(X_row) != 1:
            raise ValueError("explain_local expects exactly one row.")
        sv_in, _, output = self.shap_values(X_row)
        table = pd.DataFrame(
            {
                "feature": sv_in.columns,
                "value": [X_row.iloc[0][c] for c in sv_in.columns],
                "contribution": sv_in.iloc[0].to_numpy(),
            }
        )
        table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return {"table": table, "base_value": self.base_value, "model_output": float(output[0]), "units": self.units}


# ----------------------------------------------------------------------------- permutation importance
def permutation_importance_table(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5, max_rows: int = 20000, seed: int = 42
) -> pd.DataFrame:
    """ROC-AUC drop when each ORIGINAL input column is shuffled (fallback and cross-check for SHAP)."""
    y = np.asarray(y)
    if len(X) > max_rows:
        idx = np.random.RandomState(seed).choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    if len(np.unique(y)) < 2:
        raise ValueError("Permutation importance with ROC-AUC needs both classes in the evaluation data.")
    res = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return (
        pd.DataFrame({"feature": X.columns, "importance_mean": res.importances_mean, "importance_std": res.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
