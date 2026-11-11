"""Game Lab pipeline:  python -m src.games.pipeline

Loads the real Open Play data and the Steam Insights catalogue, runs the analyses and models, and writes
everything the dashboard reads to reports/games/ (csv/json) and data/processed/games/ (pickles, joblib).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config, project_root
from src.errors import GameSenseError
from src.games import analysis as A
from src.games import models as M
from src.games.openplay import load_open_play
from src.games.steamcat import load_catalogue, name_lookup
from src.portable import save_table
from src.utils import clean_nan, dump_json

log = logging.getLogger("gamesense.games")


def run(cfg: dict | None = None, root: Path | None = None) -> dict:
    cfg = cfg or load_config()
    root = Path(root or project_root())
    gc = cfg["games"]
    out = root / gc["out_dir"]
    proc = root / gc["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    fc = gc["frustration"]
    k = gc["min_group_participants"]
    t0 = time.time()

    log.info("loading Open Play")
    op = load_open_play(cfg, root)
    log.info("loading Steam catalogue")
    cat = load_catalogue(cfg, root)
    lookup = name_lookup(cat)

    # processed tables (local pickles, same machine only)
    save_table(op.participants, proc / "participants.pkl")
    save_table(op.diary, proc / "diary.pkl")
    save_table(op.playtime, proc / "playtime.pkl")
    save_table(op.games, proc / "games.pkl")
    cat.to_pickle(proc / "steam_catalogue.pkl")

    log.info("descriptive analysis")
    most = A.most_played(op.playtime, min_players=k)
    most.to_csv(out / "most_played.csv", index=False)
    for plat in ("Steam", "Nintendo"):
        A.most_played(op.playtime, plat, k).to_csv(out / f"most_played_{plat.lower()}.csv", index=False)
    A.genre_popularity(op.playtime).to_csv(out / "genre_popularity.csv", index=False)
    matrix = A.age_genre_lift(A.age_genre_matrix(op.playtime, op.participants, k))
    matrix.to_csv(out / "age_genre.csv", index=False)
    A.age_trend_by_genre(op.playtime, op.participants).to_csv(out / "age_genre_trend.csv", index=False)
    bands = sorted(op.participants["age_band"].dropna().unique())
    tops = pd.concat([A.top_games_by_age(op.playtime, op.participants, b, 15, k).assign(age_band=b) for b in bands])
    tops.to_csv(out / "top_games_by_age.csv", index=False)

    log.info("frustration")
    kw = dict(min_participants=fc["min_participants_per_game"], min_entries=fc["min_entries_per_game"])
    frus = A.shrunken_frustration(op.diary, **kw)
    frus["game"] = frus["game_key"].map(op.diary.groupby("game_key")["game_label"].agg(lambda s: s.value_counts().index[0]))
    frus.to_csv(out / "frustration_by_game.csv", index=False)
    delta = A.within_person_delta(op.diary, fc["min_participants_per_game"], fc["min_entries_per_game"])
    delta["game"] = delta["game_key"].map(op.diary.groupby("game_key")["game_label"].agg(lambda s: s.value_counts().index[0]))
    delta.to_csv(out / "frustration_within_person.csv", index=False)
    A.frustration_by_genre(op.diary, op.games, k, fc["min_entries_per_game"]).to_csv(out / "frustration_by_genre.csv", index=False)
    A.frustration_by_group(op.diary, "platform", k).to_csv(out / "frustration_by_platform.csv", index=False)
    A.frustration_by_age(op.diary, k).to_csv(out / "frustration_by_age.csv", index=False)
    A.need_profile(op.diary, list(frus["game_key"].head(40))).merge(frus[["game_key", "game"]], on="game_key").to_csv(out / "frustration_needs.csv", index=False)
    reliability = A.frustration_reliability(op.diary, fc["items"])
    link = A.steam_reception_link(delta, lookup)

    log.info("models")
    fm = M.frustration_model(op.diary, op.games, cfg)
    fm["metrics"].to_csv(out / "model_frustration_metrics.csv", index=False)
    fm["coefficients"].to_csv(out / "model_frustration_coefficients.csv", index=False)
    joblib.dump(fm, proc / "frustration_model.joblib")
    rm = M.reception_model(cat, cfg)
    rm["metrics"].to_csv(out / "model_reception_metrics.csv", index=False)
    rm["importance"].to_csv(out / "model_reception_importance.csv", index=False)
    joblib.dump(rm, proc / "reception_model.joblib")
    rec = M.build_recommender(op.playtime, cfg)
    rec["evaluation"]["metrics"].to_csv(out / "model_recommender_metrics.csv", index=False)
    joblib.dump(rec, proc / "recommender.joblib")
    M.save_recommender_portable(rec, proc)

    summary = {
        "load_info": op.load_info,
        "steam_catalogue_games": int(len(cat)),
        "diary_games_matched_to_telemetry": float(op.diary["game_key"].isin(op.games["game_key"]).mean()),
        "diary_games_matched_to_steam_catalogue": float(op.diary["game_key"].isin(lookup["game_key"]).mean()),
        "frustration": {"high_threshold": fc["high_threshold"], "high_rate": float(op.diary["high_frustration"].mean()),
                        "mean": float(op.diary["frustration"].mean()), **reliability, "games_ranked": int(len(frus))},
        "steam_reception_link": link,
        "models": {
            "frustration": {"prevalence": fm["prevalence"], "n_entries": fm["n_entries"], "n_participants": fm["n_participants"]},
            "reception": {k2: rm[k2] for k2 in ("n_train", "n_valid", "n_test", "positive_rate_train", "positive_rate_test")},
            "recommender": {k2: rec["evaluation"][k2] for k2 in ("n_users", "n_games", "density")},
        },
        "min_group_participants": k,
        "seconds": round(time.time() - t0, 1),
    }
    dump_json(out / "summary.json", clean_nan(summary))
    log.info("done in %.1fs", summary["seconds"])
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the GameSense Game Lab pipeline on the Open Play data.")
    ap.add_argument("--root", default=None, help="project root (default: auto)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(root=Path(args.root) if args.root else None)
    except GameSenseError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
