"""Sessionisation of poll-level records.

The WoWAH data are 10-minute polls: one row = "this avatar was online at this time".
A session is a maximal run of polls for one avatar with no gap longer than ``gap_minutes``.
Session length is an *estimate*: (last poll - first poll) + one poll interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sessionize(events: pd.DataFrame, gap_minutes: float, poll_minutes: float) -> pd.DataFrame:
    """Return one row per session: avatar, start, end, n_polls, duration_min."""
    cols = ["avatar", "start", "end", "n_polls", "duration_min"]
    if len(events) == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})

    df = events[["avatar", "timestamp"]].sort_values(["avatar", "timestamp"], kind="stable")
    av = df["avatar"].to_numpy(np.int64)
    ts = df["timestamp"].to_numpy("datetime64[ns]")

    new_session = np.ones(len(df), dtype=bool)
    if len(df) > 1:
        gap = ts[1:] - ts[:-1]
        limit = np.timedelta64(int(round(gap_minutes * 60)), "s")
        new_session[1:] = (av[1:] != av[:-1]) | (gap > limit)

    starts = np.flatnonzero(new_session)
    ends = np.append(starts[1:], len(df)) - 1
    span_min = (ts[ends] - ts[starts]) / np.timedelta64(1, "m")
    return pd.DataFrame(
        {
            "avatar": av[starts],
            "start": ts[starts],
            "end": ts[ends],
            "n_polls": (ends - starts + 1).astype(np.int64),
            "duration_min": span_min.astype(np.float64) + float(poll_minutes),
        }
    )
