"""Memory-conscious loading and validation of the raw WoWAH CSV.

Design notes
------------
* The file is read in chunks with only the needed columns.
* Header names are matched after ``strip().lower()`` because Kaggle CSV exports
  frequently carry padded headers.
* The timestamp format is *detected* (best parse rate on a sample) and then verified
  against the whole file; a wrong guess fails loudly instead of silently mis-dating rows.
* Nothing is invented: rows that cannot be used are dropped and *counted* in the
  returned load report.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import union_categoricals

from src.config import Paths
from src.errors import (
    ArtifactMissingError,
    DatasetNotFoundError,
    EmptyDatasetError,
    SchemaError,
    TimestampParseError,
)

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("avatar", "level", "timestamp")
OPTIONAL_COLUMNS = ("race", "charclass", "zone", "guild")   # absent in the lightweight WoWAH subset
CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
CATEGORICAL_COLUMNS = ("race", "charclass", "zone")


# --------------------------------------------------------------------------- locating
def download_instructions(cfg: dict[str, Any], paths: Paths) -> str:
    ds = cfg["dataset"]
    return (
        f"Dataset file not found.\n"
        f"  1. Open {ds['kaggle_url']} (free Kaggle account required) and download the data.\n"
        f"  2. Unzip it and place the CSV (expected name: '{ds['expected_file']}') in:\n"
        f"       {paths.raw_dir}\n"
        f"  3. Re-run:  python -m src.pipeline\n"
        f"Alternatively pass an explicit path:  python -m src.pipeline --raw-file <path-to-csv>"
    )


def find_raw_file(cfg: dict[str, Any], paths: Paths, raw_file: str | Path | None = None) -> Path:
    """Locate the raw CSV or raise DatasetNotFoundError with download instructions."""
    if raw_file is not None:
        p = Path(raw_file)
        if p.is_file():
            return p
        raise DatasetNotFoundError(f"Raw file not found: {p}\n\n{download_instructions(cfg, paths)}")
    expected = paths.raw_dir / cfg["dataset"]["expected_file"]
    if expected.is_file():
        return expected
    csvs = sorted(paths.raw_dir.glob("*.csv")) if paths.raw_dir.exists() else []
    if len(csvs) == 1:
        log.warning("Expected %s not found; using the only CSV in %s: %s", expected.name, paths.raw_dir, csvs[0].name)
        return csvs[0]
    if len(csvs) > 1:
        raise DatasetNotFoundError(
            f"'{expected.name}' not found and {paths.raw_dir} contains several CSV files "
            f"({', '.join(c.name for c in csvs)}). Keep one, or pass --raw-file."
        )
    raise DatasetNotFoundError(download_instructions(cfg, paths))


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Size and SHA-256 of the raw file, recorded for provenance."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return {"file_name": path.name, "file_bytes": path.stat().st_size, "sha256": sha.hexdigest()}


# --------------------------------------------------------------------------- schema
def resolve_columns(raw_columns: list[str], aliases: dict[str, list[str]]) -> dict[str, str]:
    """Map canonical column -> raw header, tolerant of padding and case."""
    normalised = {str(c).strip().lower(): c for c in raw_columns}
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for canon in CANONICAL_COLUMNS:
        candidates = aliases.get(canon, [canon])
        found = next((normalised[a.lower()] for a in candidates if a.lower() in normalised), None)
        if found is not None:
            mapping[canon] = found
        elif canon in REQUIRED_COLUMNS:
            missing.append(canon)
    if missing:
        raise SchemaError(
            f"Dataset is missing required column(s): {missing}. "
            f"Columns found: {[str(c).strip() for c in raw_columns]}. "
            f"Add alternative header names under dataset.columns in configs/config.yaml if needed."
        )
    return mapping


def detect_timestamp_format(sample: pd.Series, formats: list[str], min_rate: float) -> tuple[str | None, float]:
    """Return (format, parse_rate). format None means generic ISO-like parsing was used."""
    sample = sample.dropna().astype(str).str.strip()
    if sample.empty:
        raise TimestampParseError("The timestamp column contains no values.")
    best_fmt, best_rate = None, 0.0
    for fmt in formats:
        rate = float(pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean())
        if rate > best_rate:
            best_fmt, best_rate = fmt, rate
    if best_rate >= min_rate:
        return best_fmt, best_rate
    generic = float(pd.to_datetime(sample, errors="coerce").notna().mean())
    if generic >= min_rate:
        return None, generic
    raise TimestampParseError(
        f"Could not parse timestamps reliably (best rate {max(best_rate, generic):.1%} < {min_rate:.0%}). "
        f"Example values: {sample.head(3).tolist()}. Add the correct format to dataset.timestamp_formats."
    )


# --------------------------------------------------------------------------- loading
def _new_counters() -> dict[str, Any]:
    return {
        "rows_read": 0,
        "missing_raw": {c: 0 for c in CANONICAL_COLUMNS},
        "non_numeric": {"avatar": 0, "level": 0, "guild": 0},
        "bad_timestamp": 0,
        "dropped_unusable_avatar_or_timestamp": 0,
        "dropped_invalid_level": 0,
        "missing_guild_treated_as_no_guild": 0,
        "missing_category_labelled_unknown": {c: 0 for c in CATEGORICAL_COLUMNS},
    }


def _clean_chunk(chunk: pd.DataFrame, mapping: dict[str, str], ts_format: str | None, cfg: dict[str, Any], cnt: dict) -> pd.DataFrame:
    ds = cfg["dataset"]
    present = [c for c in CANONICAL_COLUMNS if c in mapping]
    df = chunk.rename(columns={raw: canon for canon, raw in mapping.items()})[present].copy()
    cnt["rows_read"] += len(df)
    for c in present:
        cnt["missing_raw"][c] += int(df[c].isna().sum())

    ts_raw = df["timestamp"].astype("string").str.strip()
    if ts_format:
        ts = pd.to_datetime(ts_raw, format=ts_format, errors="coerce")
    else:
        ts = pd.to_datetime(ts_raw, errors="coerce")
    cnt["bad_timestamp"] += int(ts.isna().sum() - ts_raw.isna().sum())
    df["timestamp"] = ts.astype("datetime64[ns]")

    for c in [c for c in ("avatar", "level", "guild") if c in df.columns]:
        num = pd.to_numeric(df[c], errors="coerce")
        cnt["non_numeric"][c] += int(num.isna().sum() - df[c].isna().sum())
        df[c] = num

    unusable = df["avatar"].isna() | df["timestamp"].isna()
    cnt["dropped_unusable_avatar_or_timestamp"] += int(unusable.sum())
    df = df.loc[~unusable]

    lo, hi = ds["level_range"]
    bad_level = df["level"].isna() | (df["level"] < lo) | (df["level"] > hi)
    cnt["dropped_invalid_level"] += int(bad_level.sum())
    df = df.loc[~bad_level]

    if "guild" in df.columns:
        cnt["missing_guild_treated_as_no_guild"] += int(df["guild"].isna().sum())
        df["guild"] = df["guild"].fillna(ds["no_guild_value"]).astype("int64")
    df["avatar"] = df["avatar"].astype("int64")
    df["level"] = df["level"].astype("int16")

    for c in [c for c in CATEGORICAL_COLUMNS if c in df.columns]:
        s = df[c].astype("string").str.strip()
        cnt["missing_category_labelled_unknown"][c] += int(s.isna().sum())
        df[c] = s.fillna("Unknown").astype("category")
    return df.reset_index(drop=True)


def _concat_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for col in chunks[0].columns:
        parts = [c[col] for c in chunks]
        if isinstance(parts[0].dtype, pd.CategoricalDtype):
            cols[col] = pd.Series(union_categoricals([p.array for p in parts]))
        else:
            cols[col] = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(cols)


def load_raw_events(path: str | Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, validate and clean the raw CSV. Returns (events, load_report).

    ``events`` is sorted by timestamp and has columns
    avatar(int32/int64), level(int16), race/charclass/zone(category), guild(int32/int64), timestamp(datetime64[ns]).
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetNotFoundError(f"Raw file not found: {path}")
    ds = cfg["dataset"]

    csv_kwargs = {"skipinitialspace": True}     # same parsing options for header and body
    header = pd.read_csv(path, nrows=0, **csv_kwargs)
    mapping = resolve_columns(list(header.columns), ds["columns"])
    usecols = list(mapping.values())
    dtypes = {mapping[c]: str for c in ("race", "charclass", "zone", "timestamp") if c in mapping}

    cnt = _new_counters()
    chunks: list[pd.DataFrame] = []
    ts_format: str | None = None
    ts_rate_sample = 1.0
    detected = False

    reader = pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=int(ds["chunksize"]), **csv_kwargs)
    for i, chunk in enumerate(reader):
        if len(chunk) == 0:
            continue
        if not detected:
            ts_format, ts_rate_sample = detect_timestamp_format(
                chunk[mapping["timestamp"]], ds["timestamp_formats"], ds["min_timestamp_parse_rate"]
            )
            detected = True
            log.info("Timestamp format: %s (sample parse rate %.2f%%)", ts_format or "generic", ts_rate_sample * 100)
        chunks.append(_clean_chunk(chunk, mapping, ts_format, cfg, cnt))
        log.info("Processed chunk %d (%d rows read so far)", i + 1, cnt["rows_read"])

    if cnt["rows_read"] == 0 or not chunks:
        raise EmptyDatasetError(f"{path.name} contains no data rows.")

    present_ts = cnt["rows_read"] - cnt["missing_raw"]["timestamp"]
    parse_rate = 1.0 - (cnt["bad_timestamp"] / present_ts) if present_ts else 0.0
    if parse_rate < ds["min_timestamp_parse_rate"]:
        raise TimestampParseError(
            f"Only {parse_rate:.1%} of timestamps parsed with format {ts_format!r} over the whole file "
            f"(required {ds['min_timestamp_parse_rate']:.0%}). The date format may be ambiguous "
            f"(day/month order). Set the right format first in dataset.timestamp_formats."
        )

    events = _concat_chunks(chunks)
    del chunks
    if events.empty:
        raise EmptyDatasetError("No usable rows remain after validation.")

    dup = events.duplicated(subset=["avatar", "timestamp"], keep="first")
    n_dup = int(dup.sum())
    if n_dup:
        events = events.loc[~dup]
    events = events.sort_values(["timestamp", "avatar"], kind="stable").reset_index(drop=True)
    events["avatar"] = events["avatar"].astype("int32" if events["avatar"].max() < 2**31 - 1 else "int64")
    if "guild" in events.columns:
        events["guild"] = events["guild"].astype("int32" if events["guild"].abs().max() < 2**31 - 1 else "int64")

    report = {
        "file": file_fingerprint(path),
        "raw_header": [str(c) for c in header.columns],
        "column_mapping": mapping,
        "optional_columns_present": [c for c in OPTIONAL_COLUMNS if c in mapping],
        "optional_columns_absent": [c for c in OPTIONAL_COLUMNS if c not in mapping],
        "timestamp_format": ts_format or "generic (pandas inference)",
        "timestamp_parse_rate_overall": round(parse_rate, 6),
        "rows_read": cnt["rows_read"],
        "missing_values_raw": {c: v for c, v in cnt["missing_raw"].items() if c in mapping},
        "non_numeric_values": cnt["non_numeric"],
        "unparseable_timestamps": cnt["bad_timestamp"],
        "dropped_unusable_avatar_or_timestamp": cnt["dropped_unusable_avatar_or_timestamp"],
        "dropped_invalid_level": cnt["dropped_invalid_level"],
        "level_range_applied": list(ds["level_range"]),
        "missing_guild_treated_as_no_guild": cnt["missing_guild_treated_as_no_guild"],
        "missing_category_labelled_unknown": cnt["missing_category_labelled_unknown"],
        "duplicate_avatar_timestamp_rows_removed": n_dup,
        "rows_after_cleaning": int(len(events)),
        "rows_removed_total": int(cnt["rows_read"] - len(events)),
    }
    if len(events) == 0:
        raise EmptyDatasetError("No usable rows remain after validation.")
    return events, report


# --------------------------------------------------------------------------- persistence
def save_table(df: pd.DataFrame, path: Path) -> None:
    """Persist a pandas table as a local pickle cache (fast, preserves dtypes incl. categoricals)."""
    from src.portable import save_table as _save
    _save(df, path)


def load_table(path: Path, what: str = "table") -> pd.DataFrame:
    from src.portable import read_table, table_exists
    path = Path(path)
    if not table_exists(path):
        raise ArtifactMissingError(
            f"Processed {what} not found at {path}. Generate it first with:  python -m src.pipeline"
        )
    return read_table(path)


def save_events(events: pd.DataFrame, path: Path) -> None:
    save_table(events, path)


def load_events(path: Path) -> pd.DataFrame:
    events = load_table(path, "events")
    events["timestamp"] = events["timestamp"].astype("datetime64[ns]")
    if not events["timestamp"].is_monotonic_increasing:
        events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return events
