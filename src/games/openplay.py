"""Loading and cleaning the Open Play dataset (data/clean tables).

Only participants flagged ``qualified`` in the intake survey are used (the study's own inclusion rule).
Ages in this sample are 18-43 (recruited via Prolific/Pureprofile, US and UK): conclusions about age
apply to that range only. Participant ids are the dataset's pseudonyms and are never displayed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


from src.errors import DataReadError, DatasetNotFoundError, EmptyDatasetError, SchemaError
from src.games.names import normalize

FILES = {
    "intake": "survey_intake.csv.gz",
    "daily": "survey_daily.csv.gz",
    "steam": "steam.csv.gz",
    "nintendo": "nintendo.csv.gz",
    "xbox": "xbox.csv.gz",
    "meta": "game_metadata.csv.gz",
}
REQUIRED = {
    "intake": ["pid", "qualified", "age", "gender", "country"],
    "daily": ["pid", "date", "played24hr", "bangs_1", "bangs_2", "bangs_3", "bangs_4", "bangs_5", "bangs_6",
              "most_recent_game", "most_recent_game_plat"],
    "steam": ["pid", "title_id", "minutes", "steam_id"],
    "nintendo": ["pid", "title_id", "duration"],
    "xbox": ["pid", "title_id", "duration"],
    "meta": ["original_name", "platform", "igdb_name", "match_confidence", "genres", "game_modes", "steam_pct_positive"],
}
PLATFORM_FROM_DIARY = {"Steam": "Steam", "Xbox": "Xbox", "Nintendo Switch": "Nintendo", "Playstation": "PlayStation",
                       "iOS": "iOS", "Android": "Android"}
MIN_MATCH_CONFIDENCE = 0.8
GENDER_GROUPS = {"Man": "Man", "Woman": "Woman"}      # everything else (incl. non-binary) -> "Another gender"


@dataclass
class OpenPlay:
    participants: pd.DataFrame      # pid, age, age_band, gender_group, country
    diary: pd.DataFrame             # one row per diary entry with a rated session
    playtime: pd.DataFrame          # pid x platform x game: minutes, sessions
    games: pd.DataFrame             # game_key -> name, genres, modes (unified across platforms)
    load_info: dict[str, Any]


def _read(path: Path, required: list[str], label: str, **kw) -> pd.DataFrame:
    if not path.is_file():
        if list(path.parent.glob(path.name + ".part*")):
            raise DatasetNotFoundError(
                f"{path.name} was sent in parts and has not been joined yet.\n"
                "Run:  python tools\\join_parts.py   (from the GameSense folder), then run this again."
            )
        raise DatasetNotFoundError(
            f"Open Play file not found: {path}\nPut the data/clean tables from "
            f"https://github.com/digital-wellbeing/open-play (CC0) in {path.parent} (see docs/GAMES_DATA.md)."
        )
    try:
        df = pd.read_csv(path, low_memory=False, **kw)
    except (EOFError, OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataReadError(
            f"Could not read {path.name}: {type(exc).__name__}: {exc}\n"
            "The file is probably damaged or only partly extracted. Extract the zip again (or run tools\\join_parts.py again "
            "for steam.csv.gz)."
        ) from exc
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{path.name} is missing column(s) {missing}; found {list(df.columns)[:25]}...")
    return df


def age_band_labels(edges: list[int]) -> list[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"{lo}+" if hi >= 99 else f"{lo}-{hi - 1}")
    return out


def assign_age_band(age: pd.Series, edges: list[int]) -> pd.Series:
    return pd.cut(age, bins=edges, right=False, labels=age_band_labels(edges)).astype(str).replace("nan", np.nan)


def load_participants(base: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    it = _read(base / FILES["intake"], REQUIRED["intake"], "intake")
    q = it[it["qualified"].astype(str).str.lower().isin(["true", "1"])].copy()
    q["age"] = pd.to_numeric(q["age"], errors="coerce")
    bad = q["age"].isna() | (q["age"] < 16) | (q["age"] > 90)
    q = q.loc[~bad]
    q["gender_group"] = q["gender"].map(GENDER_GROUPS).fillna("Another gender")
    q["age_band"] = assign_age_band(q["age"], cfg["games"]["age_bands"])
    if q.empty:
        raise EmptyDatasetError("No qualified participants with a valid age in the intake survey.")
    return q[["pid", "age", "age_band", "gender_group", "country"]].reset_index(drop=True)


def build_game_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (platform, original_name) with a display name, harmonised genres and modes."""
    m = meta.copy()
    ok = m["igdb_name"].notna() & (pd.to_numeric(m["match_confidence"], errors="coerce") >= MIN_MATCH_CONFIDENCE)
    m["display_name"] = np.where(ok, m["igdb_name"], np.where(m["platform"] != "Xbox", m["original_name"], None))
    m["genres_clean"] = np.where(ok, m["genres"], None)
    if "xbox_primary_genre" in m.columns:
        fallback = m["xbox_primary_genre"].where(m["platform"] == "Xbox")
        m["genres_clean"] = m["genres_clean"].where(pd.notna(m["genres_clean"]), fallback)
    m["modes"] = np.where(ok, m["game_modes"], None)
    m["pct_positive"] = pd.to_numeric(m["steam_pct_positive"], errors="coerce")
    return m[["original_name", "platform", "display_name", "genres_clean", "modes", "pct_positive"]]


def build_playtime(base: Path, participants: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    gt = build_game_table(meta)
    keep = set(participants["pid"])
    parts = []
    info: dict[str, Any] = {}
    steam = _read(base / FILES["steam"], REQUIRED["steam"], "steam", usecols=["pid", "title_id", "minutes", "steam_id"])
    steam = steam[steam["pid"].isin(keep) & (steam["minutes"] > 0)]
    g = steam.groupby(["pid", "title_id"], observed=True).agg(minutes=("minutes", "sum"), sessions=("minutes", "size")).reset_index()
    g["platform"] = "Steam"
    parts.append(g.rename(columns={"title_id": "original_name"}))
    nin = _read(base / FILES["nintendo"], REQUIRED["nintendo"], "nintendo", usecols=["pid", "title_id", "duration"])
    nin = nin[nin["pid"].isin(keep) & (nin["duration"] > 0)]
    g = nin.groupby(["pid", "title_id"], observed=True).agg(minutes=("duration", "sum"), sessions=("duration", "size")).reset_index()
    g["platform"] = "Nintendo"
    parts.append(g.rename(columns={"title_id": "original_name"}))
    xb = _read(base / FILES["xbox"], REQUIRED["xbox"], "xbox", usecols=["pid", "title_id", "duration"])
    xb = xb[xb["pid"].isin(keep) & (xb["duration"] > 0)]
    g = xb.groupby(["pid", "title_id"], observed=True).agg(minutes=("duration", "sum"), sessions=("duration", "size")).reset_index()
    g["platform"] = "Xbox"
    parts.append(g.rename(columns={"title_id": "original_name"}))
    pt = pd.concat(parts, ignore_index=True)
    pt = pt.merge(gt, on=["original_name", "platform"], how="left")
    pt["named"] = pt["display_name"].notna()
    pt["game_key"] = np.where(pt["named"], pt["display_name"].map(normalize), "unnamed:" + pt["platform"] + ":" + pt["original_name"].astype(str))
    pt["game_name"] = np.where(pt["named"], pt["display_name"], "(unnamed " + pt["platform"] + " title)")
    pt["hours"] = pt["minutes"] / 60.0
    info["playtime_rows"] = int(len(pt))
    info["share_hours_unnamed"] = float(pt.loc[~pt["named"], "hours"].sum() / pt["hours"].sum())
    info["share_hours_with_genre"] = float(pt.loc[pt["genres_clean"].notna(), "hours"].sum() / pt["hours"].sum())
    info["hours_by_platform"] = {k: float(v) for k, v in pt.groupby("platform")["hours"].sum().items()}
    return pt.drop(columns=["minutes"]), info


def frustration_columns(diary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    fc = cfg["games"]["frustration"]
    d = diary.copy()
    d["frustration"] = d[fc["items"]].mean(axis=1)
    d["satisfaction"] = d[fc["satisfaction_items"]].mean(axis=1)
    d["autonomy_frustration"] = d["bangs_2"]
    d["competence_frustration"] = d["bangs_4"]
    d["relatedness_frustration"] = d["bangs_6"]
    d["high_frustration"] = (d["frustration"] >= fc["high_threshold"]).astype(int)
    return d


def load_diary(base: Path, participants: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    dd = _read(base / FILES["daily"], REQUIRED["daily"], "daily")
    extra = [c for c in ("affective_valence", "life_sat", "wave") if c in dd.columns]
    bangs = [f"bangs_{i}" for i in range(1, 7)]
    d = dd[dd["played24hr"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
    d = d.dropna(subset=bangs)
    for c in bangs:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=bangs)
    out_of_range = ((d[bangs] < 1) | (d[bangs] > 7)).any(axis=1)
    d = d.loc[~out_of_range]
    d = d.merge(participants, on="pid", how="inner")
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    d["game_key"] = d["most_recent_game"].map(normalize)
    d["game_label"] = d["most_recent_game"].astype(str).str.strip()
    d["platform"] = d["most_recent_game_plat"].map(PLATFORM_FROM_DIARY).fillna("Other")
    d = d[d["game_key"] != ""]
    d = frustration_columns(d, cfg)
    return d[["pid", "date", "game_key", "game_label", "platform", "age", "age_band", "gender_group", "country",
              *bangs, "frustration", "satisfaction", "autonomy_frustration", "competence_frustration",
              "relatedness_frustration", "high_frustration", *extra]].sort_values(["pid", "date"]).reset_index(drop=True)


def load_open_play(cfg: dict[str, Any], root: Path | str | None = None) -> OpenPlay:
    from src.config import project_root

    base = Path(root or project_root()) / cfg["games"]["open_play_dir"]
    participants = load_participants(base, cfg)
    meta = _read(base / FILES["meta"], REQUIRED["meta"], "meta")
    playtime, info = build_playtime(base, participants, meta)
    diary = load_diary(base, participants, cfg)
    games = (
        playtime.groupby("game_key")
        .agg(game_name=("game_name", "first"), genres=("genres_clean", "first"), modes=("modes", "first"),
             named=("named", "first"), pct_positive=("pct_positive", "first"))
        .reset_index()
    )
    info.update(participants=int(len(participants)), diary_entries=int(len(diary)),
                diary_participants=int(diary["pid"].nunique()),
                playtime_participants=int(playtime["pid"].nunique()),
                age_min=int(participants["age"].min()), age_max=int(participants["age"].max()),
                age_mean=float(participants["age"].mean()))
    return OpenPlay(participants, diary, playtime, games, info)
