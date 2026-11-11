"""Loading the trained churn model and scoring validated input."""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.errors import ArtifactMissingError, PredictionInputError
from src.models.churn import _positive_proba


def load_bundle(paths) -> dict[str, Any]:
    """Load {'pipeline': fitted sklearn Pipeline, 'meta': dict}. Raises ArtifactMissingError."""
    from src.portable import table_exists
    can_rebuild = paths.churn_meta.is_file() and table_exists(paths.churn_snapshots)
    if not paths.churn_model.is_file() and not can_rebuild:
        raise ArtifactMissingError(
            f"Trained churn model not found at {paths.churn_model}. "
            f"Train it first (needs the real dataset in data/raw):  python -m src.pipeline"
        )
    try:
        if not paths.churn_model.is_file():
            raise FileNotFoundError(paths.churn_model)
        bundle = joblib.load(paths.churn_model)
    except Exception:   # saved with a different scikit-learn / numpy than the one installed now
        bundle = _load_or_rebuild(paths)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "meta" not in bundle:
        raise ArtifactMissingError(f"{paths.churn_model} is not a valid GameSense model bundle; retrain it.")
    return bundle


def rebuild_bundle(paths, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-fit the selected churn model on the saved training snapshots with the installed scikit-learn.

    Uses exactly the rows and settings of the original final fit (train + validation avatars, all non-test cutoffs).
    """
    import json

    from src.config import load_config
    from src.models.churn import _xy, build_pipeline
    from src.portable import read_table
    cfg = cfg or load_config()
    meta = json.loads(paths.churn_meta.read_text(encoding="utf-8"))
    snaps = read_table(paths.churn_snapshots)
    test_cuts = pd.to_datetime(meta["test_cutoffs"])
    refit = snaps[snaps["group"].astype(str).isin(["train", "valid"]) & ~snaps["cutoff"].isin(test_cuts)]
    X, y = _xy(refit, list(meta["numeric_features"]), list(meta["categorical_features"]))
    pipe = build_pipeline(meta["model_name"], list(meta["numeric_features"]), list(meta["categorical_features"]), cfg).fit(X, y)
    return {"pipeline": pipe, "meta": meta}


def _load_or_rebuild(paths) -> dict[str, Any]:
    local = paths.models_dir / "churn_model.rebuilt.joblib"
    if local.is_file():
        try:
            return joblib.load(local)
        except Exception:
            pass
    bundle = rebuild_bundle(paths)
    try:
        joblib.dump(bundle, local)
    except OSError:
        pass
    return bundle


def validate_features(df: pd.DataFrame, meta: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Validate user-supplied feature rows. Returns (clean_frame, warnings) or raises PredictionInputError."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise PredictionInputError("Input must be a non-empty table with one row per avatar.")
    numeric = list(meta["numeric_features"])
    categorical = list(meta["categorical_features"])
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise PredictionInputError(f"Missing required column(s): {missing}")

    out = df[numeric + categorical].copy()
    rules = meta.get("feature_rules", {})
    problems: list[str] = []
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        bad = out[c].isna() | ~np.isfinite(out[c].astype(float))
        if bad.any():
            problems.append(f"'{c}': {int(bad.sum())} missing, non-numeric or non-finite value(s)")
            continue
        lo, hi = rules.get(c, [None, None])
        if lo is not None and (out[c] < lo).any():
            problems.append(f"'{c}': value(s) below the allowed minimum {lo}")
        if hi is not None and (out[c] > hi).any():
            problems.append(f"'{c}': value(s) above the allowed maximum {hi}")

    warnings: list[str] = []
    seen = meta.get("categories_seen", {})
    for c in categorical:
        s = out[c].astype("string").str.strip()
        if s.isna().any() or (s == "").any():
            problems.append(f"'{c}': empty value(s)")
            continue
        out[c] = s.astype(str)
        unseen = sorted(set(out[c]) - set(seen.get(c, out[c])))
        if unseen:
            warnings.append(f"'{c}' contains value(s) never seen in training: {unseen}; they are treated as 'unknown'.")
    if problems:
        raise PredictionInputError("Invalid prediction input:\n- " + "\n- ".join(problems))
    return out.reset_index(drop=True), warnings


def predict_churn(bundle: dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate then score. Returns (table with churn_probability / predicted_churn, warnings)."""
    meta = bundle["meta"]
    clean, warnings = validate_features(df, meta)
    proba = _positive_proba(bundle["pipeline"], clean)
    result = pd.DataFrame(
        {
            "churn_probability": proba,
            "predicted_churn": (proba >= float(meta["threshold"])).astype(int),
        }
    )
    return result, warnings
