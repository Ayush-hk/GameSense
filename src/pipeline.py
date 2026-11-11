"""End-to-end pipeline:  python -m src.pipeline [--raw-file PATH] [--stages ingest,segment,churn,explain,report]

Stages
------
ingest   load + validate the raw CSV, write processed tables, dataset profile and dataset card
segment  behavioural features -> K-Means segmentation (artifacts + profiles)
churn    churn target, avatar/time-safe split, model comparison, trained model + metrics
explain  SHAP (if installed) or permutation importance (fallback)
report   evaluation report, insights, churn-rate tables, optional Plotly figure export

Nothing here fabricates data: if the raw file is missing the pipeline stops with download instructions.
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src.analytics.descriptive import daily_activity, dataset_profile, hourly_activity
from src.analytics.segmentation import fit_segmentation, save_segmentation
from src.config import Paths, get_paths, load_config
from src.portable import table_exists
from src.data.loader import (
    find_raw_file,
    load_events,
    load_raw_events,
    load_table,
    save_events,
    save_table,
)
from src.errors import GameSenseError
from src.explainability.explain import (
    CAUSALITY_WARNING,
    ShapExplainer,
    permutation_importance_table,
    shap_available,
)
from src.features.behavior import build_window_features
from src.features.churn_target import build_churn_dataset
from src.models.churn import train_and_evaluate
from src.models.predict import load_bundle
from src.reporting.report import (
    build_insights,
    churn_rate_by_group,
    write_dataset_card,
    write_evaluation_report,
)
from src.utils import clean_nan, dump_json, read_json

log = logging.getLogger("gamesense")

STAGES = ("ingest", "segment", "churn", "explain", "report")


# ----------------------------------------------------------------------------- stages
def stage_ingest(cfg: dict[str, Any], paths: Paths, raw_file: str | None, data_origin: str) -> pd.DataFrame:
    raw = find_raw_file(cfg, paths, raw_file)
    log.info("Loading %s", raw)
    events, load_report = load_raw_events(raw, cfg)
    save_events(events, paths.events_table)
    daily_activity(events).to_csv(paths.daily_activity, index=False)
    hourly_activity(events).to_csv(paths.hourly_activity, index=False)
    profile = dataset_profile(events, load_report, cfg)
    dump_json(paths.dataset_profile, profile)
    dump_json(paths.load_report, load_report)
    run_info = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_origin": data_origin,
        "raw_file": load_report["file"],
        "dataset": cfg["dataset"]["name"],
        "kaggle_url": cfg["dataset"]["kaggle_url"],
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "config": cfg,
    }
    dump_json(paths.run_info, run_info)
    write_dataset_card(paths, cfg, profile, load_report, run_info)
    log.info("Ingest complete: %s rows, %s avatars", f"{len(events):,}", f"{profile['avatars']:,}")
    return events


def stage_segment(cfg: dict[str, Any], paths: Paths, events: pd.DataFrame) -> None:
    start = events["timestamp"].iloc[0].floor("D")
    end = events["timestamp"].iloc[-1].floor("D") + pd.Timedelta(days=1)
    profiles = build_window_features(events, start, end, cfg, first_seen=None)
    profiles = profiles.drop(columns=["observed_tenure_days"])
    result = fit_segmentation(profiles, cfg)
    save_segmentation(result, paths)
    save_table(result["avatar_profiles"], paths.profiles_table)
    result["selection"].to_csv(paths.cluster_selection, index=False)
    result["profiles_table"].to_csv(paths.cluster_profiles, index=False)
    dump_json(paths.segmentation_report, clean_nan(result["report"]))
    log.info("Segmentation complete: k=%s", result["report"]["chosen_k"])


def stage_churn(cfg: dict[str, Any], paths: Paths, events: pd.DataFrame, data_origin: str) -> None:
    ds = build_churn_dataset(events, cfg)
    save_table(ds.snapshots, paths.churn_snapshots)
    dump_json(paths.churn_definition, clean_nan(ds.definition()))
    train_and_evaluate(ds, cfg, paths, data_origin=data_origin)
    # metrics.json may contain NaN-free floats already; make strictly JSON safe
    dump_json(paths.metrics, clean_nan(read_json(paths.metrics)))
    churn_rate_by_group(ds.snapshots).to_csv(paths.churn_by_group, index=False)
    log.info("Churn modelling complete")


def _test_frame(paths: Paths, meta: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    snaps = load_table(paths.churn_snapshots, "churn snapshots")
    test_cutoffs = pd.to_datetime(meta["test_cutoffs"])
    test = snaps[(snaps["group"] == "test") & snaps["cutoff"].isin(test_cutoffs)]
    train_like = snaps[snaps["group"] != "test"]
    feats = list(meta["numeric_features"]) + list(meta["categorical_features"])
    return test[feats], test["churned"].to_numpy(), train_like[feats]


def stage_explain(cfg: dict[str, Any], paths: Paths) -> None:
    bundle = load_bundle(paths)
    pipe, meta = bundle["pipeline"], bundle["meta"]
    X_test, y_test, X_train_like = _test_frame(paths, meta)
    ec = cfg["explain"]

    perm = permutation_importance_table(
        pipe, X_test, y_test, n_repeats=int(ec["permutation_repeats"]), max_rows=int(ec["permutation_sample"]),
        seed=int(cfg["churn"]["random_state"]),
    )
    perm.to_csv(paths.permutation_importance, index=False)

    report: dict[str, Any] = {
        "model": meta["model_name"],
        "shap_installed": shap_available(),
        "caveat": CAUSALITY_WARNING,
        "permutation_top_features": perm["feature"].head(8).tolist(),
    }
    if shap_available() and meta["model_name"] in ("random_forest", "logistic_regression"):
        bg = X_train_like.sample(n=min(500, len(X_train_like)), random_state=int(cfg["churn"]["random_state"]))
        explainer = ShapExplainer(pipe, bg, list(meta["categorical_features"]))
        g = explainer.explain_global(X_test, int(ec["shap_sample"]), int(cfg["churn"]["random_state"]))
        g["importance"].to_csv(paths.shap_global, index=False)
        g["sample"].to_csv(paths.shap_sample, index=False)
        report.update(
            {
                "method": "SHAP",
                "units": g["units"],
                "base_value": g["base_value"],
                "additivity_max_abs_error": g["additivity_max_abs_error"],
                "n_explained": g["n_explained"],
                "top_features": g["importance"]["feature"].head(8).tolist(),
                "note": f"Values are in units of {g['units']}.",
            }
        )
    else:
        reason = "the 'shap' package is not installed in this environment" if not shap_available() else "the selected model type is not supported by the SHAP wrapper"
        report.update(
            {
                "method": "permutation importance (SHAP substitute)",
                "top_features": perm["feature"].head(8).tolist(),
                "note": f"SHAP was not used because {reason}; permutation importance (drop in test ROC-AUC) is shown instead.",
            }
        )
    dump_json(paths.explain_report, clean_nan(report))
    log.info("Explainability complete (%s)", report["method"])


def stage_report(cfg: dict[str, Any], paths: Paths) -> None:
    snaps_path = paths.churn_snapshots
    if table_exists(snaps_path) and not paths.churn_by_group.is_file():
        churn_rate_by_group(load_table(snaps_path)).to_csv(paths.churn_by_group, index=False)
    write_evaluation_report(paths, cfg)
    dump_json(paths.insights, build_insights(paths))
    from src.reporting.static_figures import export_png_figures

    export_png_figures(paths)
    try:                                     # optional Plotly HTML export
        from src.reporting.figures import export_figures

        export_figures(paths)
    except ImportError:
        log.info("plotly not installed: skipping figure export")
    log.info("Report written to %s", paths.evaluation_report)


# ----------------------------------------------------------------------------- orchestration
def run_pipeline(
    cfg: dict[str, Any] | None = None,
    raw_file: str | Path | None = None,
    stages: tuple[str, ...] | list[str] = STAGES,
    root: str | Path | None = None,
    data_origin: str = "user_provided_file",
) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = get_paths(cfg, root).ensure()
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; choose from {STAGES}")
    status: dict[str, Any] = {s: {"status": "not_run"} for s in STAGES}
    prior = read_json(paths.pipeline_status)
    if prior:
        status.update({k: v for k, v in prior.items() if k in STAGES})

    events: pd.DataFrame | None = None
    for stage in STAGES:
        if stage not in stages:
            continue
        t0 = time.time()
        try:
            if stage == "ingest":
                events = stage_ingest(cfg, paths, str(raw_file) if raw_file else None, data_origin)
            else:
                if stage in ("segment", "churn") and events is None:
                    events = load_events(paths.events_table)
                if stage == "segment":
                    stage_segment(cfg, paths, events)
                elif stage == "churn":
                    stage_churn(cfg, paths, events, data_origin)
                elif stage == "explain":
                    stage_explain(cfg, paths)
                elif stage == "report":
                    stage_report(cfg, paths)
            status[stage] = {"status": "ok", "seconds": round(time.time() - t0, 1),
                             "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        except Exception as exc:                                 # record, persist, re-raise
            status[stage] = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)[:2000]}
            dump_json(paths.pipeline_status, status)
            raise
        dump_json(paths.pipeline_status, status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GameSense analysis pipeline")
    parser.add_argument("--raw-file", help="Path to the raw CSV (default: data/raw/wowah_data.csv)")
    parser.add_argument("--stages", default=",".join(STAGES), help=f"Comma-separated subset of {STAGES}")
    parser.add_argument("--config", help="Path to an alternative config.yaml")
    parser.add_argument("--root", help="Project root override")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    try:
        status = run_pipeline(cfg, raw_file=args.raw_file, stages=stages, root=args.root)
    except GameSenseError as exc:
        print(f"\nERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    for s, v in status.items():
        print(f"{s:8s} {v['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
