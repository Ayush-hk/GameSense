"""Tables that can be read by different pandas versions.

A pickle is fast but only readable by the pandas version that wrote it. Every table is therefore also saved as a
gzipped CSV plus a small JSON file that records the column types. Any pandas can read that, with no extra packages.
Reading tries the pickle first and falls back to the CSV copy when the pickle cannot be read (or is absent).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NA = "__NA__"


def csv_path(path: Path) -> Path:
    return Path(path).with_suffix(".csv.gz")


def schema_path(path: Path) -> Path:
    return Path(path).with_suffix(".schema.json")


def write_portable(df: pd.DataFrame, path: Path) -> bool:
    """Write the CSV + schema copy next to `path`. Returns False if the table has a non-default index (not supported)."""
    path = Path(path)
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0 or df.index.step != 1:
        return False
    cols = []
    for name, dt in df.dtypes.items():
        info = {"name": str(name), "dtype": str(dt)}
        if isinstance(dt, pd.CategoricalDtype):
            info["categories"] = [str(c) for c in dt.categories]
            info["ordered"] = bool(dt.ordered)
        cols.append(info)
    df.to_csv(csv_path(path), index=False, na_rep=NA, compression="gzip")
    schema_path(path).write_text(json.dumps({"columns": cols, "rows": int(len(df))}), encoding="utf-8")
    return True


def _restore(col: pd.Series, info: dict) -> pd.Series:
    dt = info["dtype"]
    if "categories" in info:
        return pd.Series(pd.Categorical(col, categories=info["categories"], ordered=info.get("ordered", False)), index=col.index)
    if dt.startswith("datetime64"):
        out = pd.to_datetime(col, utc=("UTC" in dt), format="mixed")
        try:
            return out.astype(dt)
        except (TypeError, ValueError):
            return out
    if dt in ("str", "string", "object"):
        if dt != "object" and int(pd.__version__.split(".")[0]) >= 3:
            return col.astype(dt)          # pandas 3 string type
        return col.astype(object)          # older pandas: plain Python strings
    if dt in ("bool", "boolean"):
        return col.map({"True": True, "False": False}).astype(bool if dt == "bool" else "boolean")
    return pd.to_numeric(col).astype(dt)


def read_portable(path: Path) -> pd.DataFrame:
    path = Path(path)
    schema = json.loads(schema_path(path).read_text(encoding="utf-8"))
    raw = pd.read_csv(csv_path(path), dtype=str, keep_default_na=False, na_values=[NA])
    return pd.DataFrame({c["name"]: _restore(raw[c["name"]], c) for c in schema["columns"]})


def portable_exists(path: Path) -> bool:
    return csv_path(path).is_file() and schema_path(path).is_file()


def save_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    try:
        write_portable(df, path)
    except Exception:  # the portable copy is an extra; a table that cannot be stored this way still works as a pickle
        csv_path(path).unlink(missing_ok=True)
        schema_path(path).unlink(missing_ok=True)


def table_exists(path: Path) -> bool:
    return Path(path).is_file() or portable_exists(path)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_file():
        try:
            return pd.read_pickle(path)
        except Exception:
            if not portable_exists(path):
                raise
    return read_portable(path)
