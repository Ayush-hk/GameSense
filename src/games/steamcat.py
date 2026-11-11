"""Steam catalogue (Steam Insights snapshot, Dec 2024) used for the game-reception model.

Real storefront data: review counts, SteamSpy owner ranges and playtime, release date, price, genres and tags.
The csv files use backslash-escaped quotes and ``\\N`` for missing values.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.errors import DataReadError, DatasetNotFoundError, SchemaError
from src.games.names import normalize

NEEDS = {
    "games": ["app_id", "name", "release_date", "is_free", "type"],
    "reviews": ["app_id", "positive", "negative", "total"],
    "steamspy_insights": ["app_id", "owners_range", "playtime_median_forever", "price"],
    "genres": ["app_id", "genre"],
    "tags": ["app_id", "tag"],
}
TOP_GENRES = 12
TOP_TAGS = 25


def _read_zip_csv(base: Path, name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    p = base / f"{name}.zip"
    if not p.is_file():
        raise DatasetNotFoundError(f"Steam Insights file not found: {p}\nSee docs/GAMES_DATA.md for where to get it.")
    try:
        with zipfile.ZipFile(p) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                df = pd.read_csv(fh, escapechar="\\", na_values=["N", "\\N"], usecols=usecols, low_memory=False)
    except (zipfile.BadZipFile, EOFError, OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, IndexError) as exc:
        raise DataReadError(f"Could not read {p.name}: {type(exc).__name__}: {exc}\n"
                            "The zip is probably damaged or only partly extracted. Extract it again from the original zip.") from exc
    miss = [c for c in NEEDS[name] if c not in df.columns]
    if miss:
        raise SchemaError(f"{name}.zip/{inner} is missing column(s) {miss}")
    return df


def owners_midpoint(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+)\s*\.\.\s*(\d+)").astype(float)
    return (parts[0] + parts[1]) / 2.0


def load_catalogue(cfg: dict[str, Any], root: Path | str | None = None) -> pd.DataFrame:
    """One row per Steam game with reviews, owners, playtime, price, release year, genre/tag flags."""
    from src.config import project_root

    base = Path(root or project_root()) / cfg["games"]["steam_insights_dir"]
    g = _read_zip_csv(base, "games", ["app_id", "name", "release_date", "is_free", "type"])
    g = g[g["type"].astype(str).str.lower() == "game"].copy()
    g["release_date"] = pd.to_datetime(g["release_date"], errors="coerce")
    r = _read_zip_csv(base, "reviews", ["app_id", "positive", "negative", "total"])
    s = _read_zip_csv(base, "steamspy_insights", ["app_id", "owners_range", "playtime_median_forever", "price"])
    s["owners"] = owners_midpoint(s["owners_range"])
    df = g.merge(r, on="app_id", how="left").merge(
        s[["app_id", "owners", "playtime_median_forever", "price"]], on="app_id", how="left")
    for c in ("positive", "negative", "total", "playtime_median_forever", "price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pct_positive"] = df["positive"] / (df["positive"] + df["negative"])
    df["price_usd"] = df["price"] / 100.0
    df["is_free"] = df["is_free"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    df["release_year"] = df["release_date"].dt.year
    df["game_key"] = df["name"].map(normalize)

    gen = _read_zip_csv(base, "genres")
    top = gen["genre"].value_counts().head(TOP_GENRES).index
    gx = pd.crosstab(gen["app_id"], gen["genre"]).reindex(columns=top, fill_value=0).clip(upper=1)
    gx.columns = [f"genre_{c}" for c in gx.columns]
    tag = _read_zip_csv(base, "tags")
    toptags = tag["tag"].value_counts().head(TOP_TAGS).index
    tx = pd.crosstab(tag.loc[tag["tag"].isin(toptags), "app_id"], tag.loc[tag["tag"].isin(toptags), "tag"])
    tx = tx.reindex(columns=toptags, fill_value=0).clip(upper=1)
    tx.columns = [f"tag_{c}" for c in tx.columns]
    df = df.merge(gx, left_on="app_id", right_index=True, how="left").merge(tx, left_on="app_id", right_index=True, how="left")
    flag_cols = [c for c in df.columns if c.startswith(("genre_", "tag_"))]
    df[flag_cols] = df[flag_cols].fillna(0).astype(int)
    df = df.drop(columns=["release_date", "type", "price"])
    # several app ids can share a name (demos, re-releases): keep the one with the most reviews for name lookups
    return df.sort_values("total", ascending=False).reset_index(drop=True)


def name_lookup(cat: pd.DataFrame) -> pd.DataFrame:
    """game_key -> the best-reviewed catalogue entry (used to attach Steam reception to diary games)."""
    return cat.dropna(subset=["total"]).drop_duplicates("game_key")[["game_key", "app_id", "pct_positive", "total", "owners"]]
