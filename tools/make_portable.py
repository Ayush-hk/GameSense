"""Create the version-independent copies of the saved tables and the recommender.

Run this once with the same Python that wrote the files (the one that ran the pipelines):
    python tools/make_portable.py
After that, the dashboard also works with other pandas / scikit-learn versions.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.games import models as M  # noqa: E402
from src.portable import save_table  # noqa: E402

TABLES = ["participants", "diary", "playtime", "games"]


def main() -> None:
    games = ROOT / "data" / "processed" / "games"
    for name in TABLES:
        p = games / f"{name}.pkl"
        if p.is_file():
            save_table(pd.read_pickle(p), p)
            print("portable copy:", p.name)
    for p in [ROOT / "data" / "processed" / "avatar_profiles.pkl", ROOT / "data" / "processed" / "churn_snapshots.pkl"]:
        if p.is_file():
            save_table(pd.read_pickle(p), p)
            print("portable copy:", p.name)
    rec = games / "recommender.joblib"
    if rec.is_file():
        M.save_recommender_portable(joblib.load(rec), games)
        print("portable copy: recommender.npz")


if __name__ == "__main__":
    main()
