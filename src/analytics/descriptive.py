"""Descriptive dataset analytics: profile, daily and hourly activity aggregates.

These small aggregates are written to disk so the dashboard never has to load the
10.8M-row event table.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def daily_activity(events: pd.DataFrame) -> pd.DataFrame:
    date = events["timestamp"].dt.floor("D")
    g = events.groupby(date, observed=True)
    out = pd.DataFrame({"polls": g.size(), "active_avatars": g["avatar"].nunique()}).reset_index()
    out.columns = ["date", "polls", "active_avatars"]
    return out


def hourly_activity(events: pd.DataFrame) -> pd.DataFrame:
    ts = events["timestamp"]
    out = (
        pd.DataFrame({"weekday": ts.dt.dayofweek.to_numpy(), "hour": ts.dt.hour.to_numpy()})
        .groupby(["weekday", "hour"])
        .size()
        .rename("polls")
        .reset_index()
    )
    return out


def dataset_profile(events: pd.DataFrame, load_report: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Facts measured on the cleaned data. Every number here is computed, none is assumed."""
    ts = events["timestamp"]
    per_avatar = events.groupby("avatar", observed=True).size()
    no_guild = cfg["dataset"]["no_guild_value"]
    columns = []
    for c in events.columns:
        columns.append(
            {
                "name": c,
                "dtype": str(events[c].dtype),
                "missing": int(events[c].isna().sum()),
                "n_unique": int(events[c].nunique()),
            }
        )
    return {
        "rows": int(len(events)),
        "avatars": int(per_avatar.shape[0]),
        "date_min": str(ts.min()),
        "date_max": str(ts.max()),
        "span_days": float((ts.max() - ts.min()) / pd.Timedelta(days=1)),
        "distinct_days": int(ts.dt.floor("D").nunique()),
        "level_min": int(events["level"].min()),
        "level_max": int(events["level"].max()),
        "polls_per_avatar_median": float(per_avatar.median()),
        "polls_per_avatar_max": int(per_avatar.max()),
        "zones": int(events["zone"].nunique()) if "zone" in events else None,
        "races": sorted(events["race"].astype(str).unique().tolist()) if "race" in events else None,
        "classes": sorted(events["charclass"].astype(str).unique().tolist()) if "charclass" in events else None,
        "share_rows_without_guild": float((events["guild"] == no_guild).mean()) if "guild" in events else None,
        "memory_mb": float(events.memory_usage(deep=True).sum() / 1e6),
        "columns": columns,
        "cleaning": {
            k: load_report.get(k)
            for k in (
                "rows_read",
                "rows_after_cleaning",
                "rows_removed_total",
                "duplicate_avatar_timestamp_rows_removed",
                "dropped_invalid_level",
                "dropped_unusable_avatar_or_timestamp",
                "missing_values_raw",
                "non_numeric_values",
                "timestamp_format",
                "timestamp_parse_rate_overall",
            )
        },
    }
