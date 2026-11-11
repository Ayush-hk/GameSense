"""SYNTHETIC TEST FIXTURES - used ONLY by the automated tests.

These generators create small CSV files shaped like the WoWAH data so the code can be unit-tested
without the real dataset. They are NOT the Kaggle data, are never read by the Streamlit app, and
any metric computed from them must never be reported as a result on real data.
The pipeline run in tests is tagged data_origin="synthetic_test_fixture" so generated reports say so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RACES = ["Orc", "Tauren", "Troll", "Undead", "Blood Elf"]
CLASSES = ["Warrior", "Mage", "Priest", "Rogue", "Hunter"]
ZONES = [f"Zone{i}" for i in range(12)]


def make_events_frame(n_avatars: int = 220, days: int = 240, seed: int = 0, start: str = "2008-01-01") -> pd.DataFrame:
    """Poll-level frame: one row per (avatar, 10-minute poll)."""
    rng = np.random.RandomState(seed)
    t0 = pd.Timestamp(start)
    frames = []
    for a in range(1, n_avatars + 1):
        join = int(rng.randint(0, days // 2))
        lifetime = int(rng.choice([15, 40, 80, days], p=[0.25, 0.25, 0.25, 0.25]))
        last_day = min(join + lifetime, days - 1)
        p_active = rng.uniform(0.15, 0.9)
        level = int(rng.randint(1, 20))
        race, klass = RACES[rng.randint(len(RACES))], CLASSES[rng.randint(len(CLASSES))]
        guild = int(rng.randint(1, 30)) if rng.rand() < 0.7 else -1
        rows_ts, rows_level, rows_zone = [], [], []
        for d in range(join, last_day + 1):
            if rng.rand() > p_active:
                continue
            n_sessions = int(rng.choice([1, 1, 2]))
            for _ in range(n_sessions):
                hour = int(rng.randint(0, 22))
                start_poll = t0 + pd.Timedelta(days=d, hours=hour, minutes=int(rng.randint(0, 10)))
                n_polls = int(rng.randint(2, 18))
                for k in range(n_polls):
                    rows_ts.append(start_poll + pd.Timedelta(minutes=10 * k))
                    rows_level.append(min(level, 80))
                    rows_zone.append(ZONES[rng.randint(len(ZONES))])
            if rng.rand() < 0.05:
                level += 1
        if rows_ts:
            frames.append(
                pd.DataFrame(
                    {
                        "char": a,
                        "level": rows_level,
                        "race": race,
                        "charclass": klass,
                        "zone": rows_zone,
                        "guild": guild,
                        "timestamp": rows_ts,
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def write_events_csv(path: Path, df: pd.DataFrame, fmt: str = "%m/%d/%y %H:%M:%S", pad_header: bool = True) -> Path:
    """Write like the Kaggle export: padded header names and a US-style timestamp."""
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime(fmt)
    if pad_header:
        out.columns = [out.columns[0]] + [f" {c}" for c in out.columns[1:]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def tiny_frame() -> pd.DataFrame:
    """Hand-built frame with known answers for exact assertions."""
    t = pd.Timestamp("2008-01-01 10:00:00")
    rows = []
    # avatar 1: one session of 4 polls (10:00-10:30), then a second session next day of 2 polls
    for k in range(4):
        rows.append((1, 10, "Orc", "Mage", "A", 5, t + pd.Timedelta(minutes=10 * k)))
    for k in range(2):
        rows.append((1, 11, "Orc", "Mage", "B", 5, t + pd.Timedelta(days=1, minutes=10 * k)))
    # avatar 2: two polls with a 60 minute gap -> two sessions of one poll each (gap 60 > 30)
    rows.append((2, 5, "Troll", "Rogue", "A", -1, t))
    rows.append((2, 5, "Troll", "Rogue", "A", -1, t + pd.Timedelta(minutes=60)))
    df = pd.DataFrame(rows, columns=["avatar", "level", "race", "charclass", "zone", "guild", "timestamp"])
    for c in ("race", "charclass", "zone"):
        df[c] = df[c].astype("category")
    df["avatar"] = df["avatar"].astype("int32")
    df["level"] = df["level"].astype("int16")
    df["guild"] = df["guild"].astype("int32")
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
