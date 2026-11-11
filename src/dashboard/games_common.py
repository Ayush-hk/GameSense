"""Shared helpers for the Game Lab pages. Everything shown is read from files written by src.games.pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import load_config, project_root
from src.portable import read_table, table_exists
from src.dashboard.common import LOAD_ERRORS, _mt, explain_load_failure, setup
from src.utils import read_json

NO_GAMES_HELP = """
**No Game Lab results yet.** These pages only show measured results from real data.

1. Put the Open Play tables in `data/raw_games/open_play/` and the Steam Insights zips in `data/raw_games/steam_insights/`
   (see `docs/GAMES_DATA.md` for the links).
2. Run `python -m src.games.pipeline` in a terminal, then refresh this page.
"""

WRONG_FOLDER = """
**The result summaries were found, but not the data files they were built from.**

Missing: `{path}`

GameSense is running from `{root}`. This usually means the app was started from a different copy of the
project than the one where the pipelines were run.

- Stop the app (Ctrl+C), go to the project folder that contains `data\\processed` (for example `D:\\1_GameSense_code\\GameSense`),
  activate its `.venv` and run `streamlit run app.py` there, **or**
- run `python -m src.games.pipeline` in this folder (needs the raw data in `data\\raw_games`).
"""

PRIVACY_NOTE = "Group-level results only: a group needs at least {k} participants to appear."


def games_dirs(cfg) -> tuple[Path, Path]:
    root = project_root()
    return root / cfg["games"]["out_dir"], root / cfg["games"]["processed_dir"]


def setup_games(title: str, subtitle: str = ""):
    cfg = load_config()
    out, proc = games_dirs(cfg)
    summary = read_json(out / "summary.json")
    info = (summary or {}).get("load_info", {})
    chips = ()
    if info:
        chips = (("Participants", f"{info['participants']:,}"),
                 ("Ages", f"{info['age_min']} - {info['age_max']} (mean {info['age_mean']:.1f})"))
    cfg, _paths = setup(title, subtitle, "Game Lab", chips)
    if summary is None:
        st.info(NO_GAMES_HELP)
        st.stop()
    needed = proc / "playtime.pkl"
    if not table_exists(needed):
        st.error(WRONG_FOLDER.format(path=needed, root=project_root()))
        st.stop()
    st.caption(PRIVACY_NOTE.format(k=cfg["games"]["min_group_participants"]))
    return cfg, out, proc, summary


@st.cache_data(show_spinner=False)
def _pkl(path: str, mtime: float) -> pd.DataFrame:
    return read_table(Path(path))


@st.cache_data(show_spinner=False)
def _csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def _joblib(path: str, mtime: float):
    return joblib.load(path)


def pkl(p: Path) -> pd.DataFrame:
    try:
        return _pkl(str(p), _mt(p))
    except LOAD_ERRORS as exc:
        explain_load_failure(p, exc)


def csv(p: Path) -> pd.DataFrame:
    return _csv(str(p), p.stat().st_mtime)


def jl(p: Path):
    try:
        return _joblib(str(p), p.stat().st_mtime)
    except LOAD_ERRORS as exc:
        explain_load_failure(p, exc)


@st.cache_resource(show_spinner=False)
def _recommender(path: str, mtime: float):
    from src.games import models as M
    return M.load_recommender(Path(path).parent)


def recommender(proc: Path):
    """The recommender: the version-independent files if present, otherwise the older joblib file."""
    f = proc / "recommender.npz"
    try:
        return _recommender(str(f), f.stat().st_mtime) if f.is_file() else jl(proc / "recommender.joblib")
    except LOAD_ERRORS as exc:
        explain_load_failure(f, exc)
