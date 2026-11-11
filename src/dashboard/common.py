"""Shared dashboard helpers. All content comes from files written by the pipeline."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_paths, load_config
from src.dashboard import ui
from src.models.predict import load_bundle
from src.utils import read_json

NO_DATA_HELP = """
**No analysis results found yet.** The dashboard only displays measured results.

1. Download the dataset from <https://www.kaggle.com/datasets/mylesoneill/warcraft-avatar-history>
2. Put `wowah_data.csv` in `data/raw/`
3. Run `python -m src.pipeline` in a terminal, then refresh this page.
"""


SUBTITLES = {
    "Data Explorer": "Inspect the telemetry: schema, quality checks, activity over time and avatar distributions.",
    "Player Segmentation": "Groups of avatars with similar behaviour, found with K-Means clustering.",
    "Prediction Lab": "Churn model results, single-avatar scoring and batch scoring from a CSV.",
    "Explainability": "Which behaviours drive the churn predictions.",
    "Insights & Reports": "Findings and the evaluation report, generated from the latest run.",
}


def setup(title: str, subtitle: str = "", badge: str = "", chips=()):
    st.set_page_config(page_title=f"GameSense - {title}", page_icon=":video_game:", layout="wide")
    ui.apply_profile()
    ui.hero(title, subtitle or SUBTITLES.get(title, ""), badge, chips)
    ctx = get_ctx()
    return ctx


@st.cache_resource
def get_ctx():
    cfg = load_config()
    return cfg, get_paths(cfg)


def origin_banner(paths):
    info = read_json(paths.run_info)
    if not info:
        return None
    if info.get("data_origin") != "user_provided_file":
        st.error(f"These results come from `{info.get('data_origin')}`, NOT from the Kaggle dataset. Do not cite them as real-data results.")
    return info


@st.cache_data(show_spinner=False)
def cached_json(path: str, mtime: float):
    return read_json(__import__("pathlib").Path(path))


@st.cache_data(show_spinner=False)
def cached_csv(path: str, mtime: float) -> pd.DataFrame | None:
    p = __import__("pathlib").Path(path)
    return pd.read_csv(p) if p.is_file() else None


@st.cache_data(show_spinner=False)
def cached_pickle(path: str, mtime: float) -> pd.DataFrame | None:
    from src.portable import read_table, table_exists
    p = __import__("pathlib").Path(path)
    return read_table(p) if table_exists(p) else None


def _mt(p) -> float:
    """Modification time of a file (or of whichever copy of a saved table exists); used as the cache key."""
    from src.portable import csv_path
    times = [q.stat().st_mtime for q in (p, csv_path(p)) if q.is_file()]
    return max(times) if times else 0.0


def jget(p):
    return cached_json(str(p), _mt(p))


def cget(p):
    return cached_csv(str(p), _mt(p))


LOAD_ERRORS = (TypeError, AttributeError, ImportError, ValueError, KeyError, EOFError, __import__("pickle").UnpicklingError)

ENV_HELP = """
**GameSense cannot read its saved data with the Python packages that are running right now.**

- Python in use: `{python}`
- pandas in use: `{pandas}`  

This almost always means the app was started from a different Python than the project's own `.venv`.

1. Stop the app (Ctrl+C in the terminal).
2. In the project folder run:  `.venv\\Scripts\\Activate.ps1`  then  `pip install -r requirements.txt`
3. Start again with  `python -m streamlit run app.py`  (or double-click `run_gamesense.bat`).

Technical detail: `{detail}`
"""


def explain_load_failure(path, exc):
    import sys
    st.error(ENV_HELP.format(python=sys.executable, pandas=pd.__version__, detail=f"{type(exc).__name__}: {exc} while reading {path}"))
    st.stop()


def pget(p):
    try:
        return cached_pickle(str(p), _mt(p))
    except LOAD_ERRORS as exc:
        explain_load_failure(p, exc)


@st.cache_resource
def cached_bundle(path: str, mtime: float):
    cfg = load_config()
    return load_bundle(get_paths(cfg))


def get_bundle(paths):
    try:
        return cached_bundle(str(paths.churn_model), _mt(paths.churn_model))
    except LOAD_ERRORS as exc:
        explain_load_failure(paths.churn_model, exc)


def require(condition: bool, message: str = NO_DATA_HELP):
    if not condition:
        st.info(message)
        from src.config import project_root
        st.caption(f"GameSense is running from: {project_root()}  - if you already ran the pipeline, start the app from that project folder.")
        st.stop()
