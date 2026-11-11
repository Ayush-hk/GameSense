"""Per-avatar behavioural features computed over a time window.

``build_window_features`` is the single feature engine used for BOTH segmentation
(window = whole data span) and churn prediction (window = observation window before a
cutoff). It only ever reads rows with ``start <= timestamp < end``, which is what makes
the leakage guarantees in ``churn_target`` possible.

All aggregation is vectorised NumPy over integer codes, so the 10.8M-row full dataset
fits comfortably in 16 GB RAM.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.sessions import sessionize

FEATURE_COLUMNS = [
    "avatar",
    "active_days",
    "active_day_ratio",
    "activity_regularity",
    "active_span_days",
    "n_polls",
    "est_hours_online",
    "n_sessions",
    "mean_session_min",
    "max_session_min",
    "sessions_per_active_day",
    "hours_per_active_day",
    "days_since_last_active",
    "active_days_last7",
    "active_days_prev7",
    "level_first",
    "level_last",
    "level_gain",
    "zones_visited",
    "has_guild",
    "guild_changes",
    "weekend_share",
    "night_share",
    "observed_tenure_days",
    "race",
    "charclass",
]

FEATURE_LABELS = {
    "active_days": "Active days",
    "active_day_ratio": "Share of window days active",
    "activity_regularity": "Activity regularity (active days / lifetime span)",
    "active_span_days": "Span between first and last activity (days)",
    "n_polls": "Number of polls observed",
    "est_hours_online": "Estimated hours online",
    "n_sessions": "Number of sessions",
    "mean_session_min": "Mean session length (min)",
    "max_session_min": "Longest session (min)",
    "sessions_per_active_day": "Sessions per active day",
    "hours_per_active_day": "Hours per active day",
    "days_since_last_active": "Days since last activity",
    "active_days_last7": "Active days in last 7 days",
    "active_days_prev7": "Active days in the 7 days before that",
    "level_first": "Level at start of window",
    "level_last": "Level at end of window",
    "level_gain": "Levels gained in window",
    "zones_visited": "Distinct zones visited",
    "has_guild": "In a guild at end of window",
    "guild_changes": "Guild changes in window",
    "weekend_share": "Share of activity on weekends",
    "night_share": "Share of activity 00:00-05:59",
    "observed_tenure_days": "Observed tenure (days, left-censored)",
    "race": "Race",
    "charclass": "Class",
}


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in FEATURE_COLUMNS})


def _nunique_per_group(inv: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Number of distinct ``values`` for each group code in ``inv``."""
    if inv.size == 0:
        return np.zeros(n_groups, dtype=np.int64)
    v = values.astype(np.int64)
    v = v - v.min()
    span = int(v.max()) + 1
    key = np.unique(inv.astype(np.int64) * span + v)
    return np.bincount(key // span, minlength=n_groups)


def compute_first_seen(events: pd.DataFrame) -> pd.Series:
    """First timestamp each avatar appears anywhere in the data (index = avatar).

    Safe to use for an avatar that is active before a cutoff: its first appearance is
    necessarily before that cutoff, so no future information is used.
    """
    return events.groupby("avatar", observed=True)["timestamp"].min()


def build_window_features(
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: dict[str, Any],
    first_seen: pd.Series | None = None,
) -> pd.DataFrame:
    """Features for every avatar with at least one poll in ``[start, end)``.

    ``events`` must be sorted by timestamp (``load_events`` guarantees this).
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError("Window end must be after window start.")
    poll = float(cfg["dataset"]["poll_minutes"])
    gap = float(cfg["sessions"]["gap_minutes"])
    no_guild = cfg["dataset"]["no_guild_value"]

    ts_all = events["timestamp"].to_numpy("datetime64[ns]")
    lo = int(np.searchsorted(ts_all, start.to_datetime64(), side="left"))
    hi = int(np.searchsorted(ts_all, end.to_datetime64(), side="left"))
    w = events.iloc[lo:hi]
    if len(w) == 0:
        return _empty_features()

    av = w["avatar"].to_numpy(np.int64)
    ts = w["timestamp"].to_numpy("datetime64[ns]")
    uniq, inv = np.unique(av, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(uniq)

    # Deterministic first/last poll per avatar. `events` is time-sorted, and a *stable* sort by
    # avatar code keeps time order inside each avatar block, so block start/end = first/last poll.
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=n)
    block_end = np.cumsum(counts) - 1
    block_start = block_end - counts + 1
    first_idx = order[block_start]
    last_idx = order[block_end]

    one_day = np.timedelta64(1, "D")
    window_days = max((end - start) / pd.Timedelta(days=1), 1e-9)
    day_idx = ((ts - start.to_datetime64()) // one_day).astype(np.int64)
    days_to_end = ((end.to_datetime64() - ts) // one_day).astype(np.int64)

    n_polls = np.bincount(inv, minlength=n)
    active_days = _nunique_per_group(inv, day_idx, n)
    last7_mask = days_to_end < 7
    prev7_mask = (days_to_end >= 7) & (days_to_end < 14)
    active_last7 = _nunique_per_group(inv[last7_mask], day_idx[last7_mask], n)
    active_prev7 = _nunique_per_group(inv[prev7_mask], day_idx[prev7_mask], n)

    has_zone, has_guild_col = "zone" in w.columns, "guild" in w.columns
    if has_zone:
        zone_codes = w["zone"].cat.codes.to_numpy(np.int64) if isinstance(w["zone"].dtype, pd.CategoricalDtype) else pd.factorize(w["zone"])[0]
        zones_visited = _nunique_per_group(inv, zone_codes, n)
    if has_guild_col:
        guild = w["guild"].to_numpy(np.int64)
        guild_distinct = _nunique_per_group(inv, guild, n)
        has_guild = (guild[last_idx] != no_guild).astype(np.int8)
        guild_changes = np.maximum(guild_distinct - 1, 0)

    level = w["level"].to_numpy(np.int64)
    level_first = level[first_idx]
    level_last = level[last_idx]

    # sessions (computed strictly inside the window -> no session straddles the cutoff)
    sess = sessionize(w, gap, poll)
    s_inv = np.searchsorted(uniq, sess["avatar"].to_numpy(np.int64))
    n_sessions = np.bincount(s_inv, minlength=n)
    dur = sess["duration_min"].to_numpy(np.float64)
    sum_dur = np.bincount(s_inv, weights=dur, minlength=n)
    max_dur = np.zeros(n, dtype=np.float64)
    np.maximum.at(max_dur, s_inv, dur)

    # temporal habits
    days_since_epoch = ts.astype("datetime64[D]").astype(np.int64)
    weekday = (days_since_epoch + 3) % 7           # 1970-01-01 was a Thursday -> Monday = 0
    hour = ((ts - ts.astype("datetime64[D]")) // np.timedelta64(1, "h")).astype(np.int64)
    weekend_share = np.bincount(inv, weights=(weekday >= 5).astype(float), minlength=n) / n_polls
    night_share = np.bincount(inv, weights=(hour < 6).astype(float), minlength=n) / n_polls

    last_ts = ts[last_idx]
    first_ts = ts[first_idx]
    span_days = (last_ts - first_ts) / one_day
    est_hours = n_polls * poll / 60.0


    out = pd.DataFrame(
        {
            "avatar": uniq,
            "active_days": active_days,
            "active_day_ratio": active_days / window_days,
            "activity_regularity": active_days / (span_days + 1.0),
            "active_span_days": span_days,
            "n_polls": n_polls,
            "est_hours_online": est_hours,
            "n_sessions": n_sessions,
            "mean_session_min": sum_dur / np.maximum(n_sessions, 1),
            "max_session_min": max_dur,
            "sessions_per_active_day": n_sessions / np.maximum(active_days, 1),
            "hours_per_active_day": est_hours / np.maximum(active_days, 1),
            "days_since_last_active": (end.to_datetime64() - last_ts) / one_day,
            "active_days_last7": active_last7,
            "active_days_prev7": active_prev7,
            "level_first": level_first,
            "level_last": level_last,
            "level_gain": np.maximum(level_last - level_first, 0),
            "weekend_share": weekend_share,
            "night_share": night_share,
        }
    )
    # optional columns: only present when the source file provides them
    if has_zone:
        out["zones_visited"] = zones_visited
    if has_guild_col:
        out["has_guild"] = has_guild
        out["guild_changes"] = guild_changes
    for cat in ("race", "charclass"):
        if cat in w.columns:
            out[cat] = w[cat].astype(str).to_numpy()[last_idx]
    if first_seen is not None:
        fs = first_seen.reindex(uniq).to_numpy("datetime64[ns]")
        out["observed_tenure_days"] = (end.to_datetime64() - fs) / one_day
    else:
        out["observed_tenure_days"] = np.nan
    return out[[c for c in FEATURE_COLUMNS if c in out.columns]]
