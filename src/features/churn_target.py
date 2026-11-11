"""Churn target construction with explicit windows and leakage safeguards.

Definition (all values configurable in configs/config.yaml)
-----------------------------------------------------------
For a *cutoff* date C:

* **Observation window**  [C - observation_days, C)  - the only data used for features.
* **Prediction window**   [C, C + prediction_days)    - the only data used for the label.
* **Population**          avatars with at least ``min_active_days`` active days in the observation window.
* **Churn label**         1 if the avatar has ZERO polls in the whole prediction window
                          (inactivity threshold = prediction_days), else 0.

Safeguards
----------
1. Feature window end <= label window start (``assert_no_leakage``; raises ``LeakageError``).
2. A cutoff is only allowed if its whole prediction window lies inside the observed data
   (no label is ever computed from incomplete follow-up).
3. Out-of-time evaluation: test cutoff is the newest possible; training cutoffs are older and
   their label windows end no later than the test cutoff.
4. Avatar-disjoint evaluation: avatars are split into train / validation / test groups, so the
   same avatar can never appear in both fitting and evaluation data, at any cutoff.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.errors import InsufficientHistoryError, LeakageError
from src.features.behavior import build_window_features, compute_first_seen


@dataclass(frozen=True)
class ChurnSpec:
    observation_days: int = 28
    prediction_days: int = 28
    n_test_cutoffs: int = 3
    n_train_cutoffs: int = 24
    n_valid_cutoffs: int = 4
    min_active_days: int = 1
    valid_avatar_fraction: float = 0.15
    test_avatar_fraction: float = 0.25
    random_state: int = 42

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ChurnSpec":
        c = cfg["churn"]
        return cls(
            observation_days=int(c["observation_days"]),
            prediction_days=int(c["prediction_days"]),
            n_test_cutoffs=int(c.get("n_test_cutoffs", 3)),
            n_train_cutoffs=int(c["n_train_cutoffs"]),
            n_valid_cutoffs=int(c.get("n_valid_cutoffs", 4)),
            min_active_days=int(c["min_active_days"]),
            valid_avatar_fraction=float(c["valid_avatar_fraction"]),
            test_avatar_fraction=float(c["test_avatar_fraction"]),
            random_state=int(c["random_state"]),
        )


@dataclass
class CutoffPlan:
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    test_cutoffs: list[pd.Timestamp]           # newest first
    train_cutoffs: list[pd.Timestamp]          # newest first: validation cutoffs, then fitting cutoffs

    @property
    def test_cutoff(self) -> pd.Timestamp:     # the newest test cutoff
        return self.test_cutoffs[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable_data_start": str(self.data_start),
            "usable_data_end": str(self.data_end),
            "test_cutoff": str(self.test_cutoff),
            "test_cutoffs_newest_first": [str(c) for c in self.test_cutoffs],
            "train_cutoffs_newest_first": [str(c) for c in self.train_cutoffs],
        }


def assert_no_leakage(feature_window_end: pd.Timestamp, label_window_start: pd.Timestamp) -> None:
    if pd.Timestamp(feature_window_end) > pd.Timestamp(label_window_start):
        raise LeakageError(
            f"Feature window ends at {feature_window_end} which is after the label window "
            f"starts at {label_window_start}: temporal leakage."
        )


def plan_cutoffs(t_min: pd.Timestamp, t_max: pd.Timestamp, spec: ChurnSpec) -> CutoffPlan:
    """Choose the test cutoff and the older training cutoffs from the available data span."""
    t_min, t_max = pd.Timestamp(t_min), pd.Timestamp(t_max)
    data_start = t_min if t_min == t_min.floor("D") else t_min.ceil("D")
    data_end = t_max.floor("D")                       # drop the possibly partial last day
    obs = pd.Timedelta(days=spec.observation_days)
    pred = pd.Timedelta(days=spec.prediction_days)

    tests = [data_end - (k + 1) * pred for k in range(spec.n_test_cutoffs)]       # newest first
    if tests[-1] - obs < data_start:
        raise InsufficientHistoryError(
            f"Data span {data_start.date()} -> {data_end.date()} is too short for {spec.n_test_cutoffs} test period(s) of "
            f"{spec.prediction_days} d with a {spec.observation_days} d observation window."
        )
    # Training labels must be fully known BEFORE the earliest test observation window starts.
    first_train = tests[-1] - obs - pred
    train: list[pd.Timestamp] = []
    for k in range(spec.n_train_cutoffs):
        c = first_train - k * pred
        if c - obs < data_start:
            break
        train.append(c)
    if len(train) < spec.n_valid_cutoffs + 1:
        need = spec.observation_days + (spec.n_valid_cutoffs + 2) * spec.prediction_days
        raise InsufficientHistoryError(
            f"Need at least {spec.n_valid_cutoffs + 1} training cutoffs ({spec.n_valid_cutoffs} for validation, 1+ for fitting) but the data span "
            f"{data_start.date()} -> {data_end.date()} only allows {len(train)}. "
            f"At least {need} days of data are required with the current windows; "
            f"shorten churn.observation_days / churn.prediction_days in configs/config.yaml."
        )
    return CutoffPlan(data_start, data_end, tests, train)


def split_avatar_groups(avatars: np.ndarray, spec: ChurnSpec) -> dict[str, np.ndarray]:
    """Deterministic, disjoint train / valid / test split of avatar ids."""
    uniq = np.unique(avatars)
    rng = np.random.RandomState(spec.random_state)
    perm = rng.permutation(uniq)
    n = len(perm)
    n_test = int(round(n * spec.test_avatar_fraction))
    n_valid = int(round(n * spec.valid_avatar_fraction))
    return {
        "test": np.sort(perm[:n_test]),
        "valid": np.sort(perm[n_test : n_test + n_valid]),
        "train": np.sort(perm[n_test + n_valid :]),
    }


def _active_avatars(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    ts_all = events["timestamp"].to_numpy("datetime64[ns]")
    lo = int(np.searchsorted(ts_all, start.to_datetime64(), side="left"))
    hi = int(np.searchsorted(ts_all, end.to_datetime64(), side="left"))
    return np.unique(events["avatar"].to_numpy()[lo:hi])


def build_snapshot(
    events: pd.DataFrame,
    cutoff: pd.Timestamp,
    spec: ChurnSpec,
    cfg: dict[str, Any],
    first_seen: pd.Series,
    data_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Features (observation window) + churn label (prediction window) at one cutoff."""
    cutoff = pd.Timestamp(cutoff)
    feat_start = cutoff - pd.Timedelta(days=spec.observation_days)
    label_end = cutoff + pd.Timedelta(days=spec.prediction_days)
    assert_no_leakage(cutoff, cutoff)               # feature end == cutoff == label start

    if data_end is not None and label_end > pd.Timestamp(data_end):
        raise InsufficientHistoryError(
            f"Prediction window ends {label_end} but usable data ends {data_end}: insufficient follow-up."
        )
    feats = build_window_features(events, feat_start, cutoff, cfg, first_seen)
    feats = feats.loc[feats["active_days"] >= spec.min_active_days].copy()
    future = _active_avatars(events, cutoff, label_end)
    feats["churned"] = (~np.isin(feats["avatar"].to_numpy(), future)).astype(np.int8)
    feats["cutoff"] = cutoff
    return feats.reset_index(drop=True)


@dataclass
class ChurnDataset:
    spec: ChurnSpec
    plan: CutoffPlan
    snapshots: pd.DataFrame          # every snapshot row with `group` and `cutoff`
    fit: pd.DataFrame                # train avatars @ older cutoffs C_2..C_n  (model fitting during selection)
    valid: pd.DataFrame              # valid avatars @ C_1                       (model selection / threshold)
    refit: pd.DataFrame              # train+valid avatars @ C_1..C_n            (final fit)
    test: pd.DataFrame               # test avatars @ test cutoffs               (final out-of-time evaluation)

    def definition(self) -> dict[str, Any]:
        def prev(df: pd.DataFrame) -> dict[str, Any]:
            return {"rows": int(len(df)), "churn_rate": float(df["churned"].mean()) if len(df) else None}

        per_cutoff = (
            self.snapshots.groupby(["cutoff", "group"], observed=True)["churned"]
            .agg(rows="size", churn_rate="mean")
            .reset_index()
        )
        per_cutoff["cutoff"] = per_cutoff["cutoff"].astype(str)
        return {
            "definition": (
                f"Churn = zero polls in the {self.spec.prediction_days}-day prediction window following a cutoff, "
                f"for avatars active (>= {self.spec.min_active_days} active day) in the preceding "
                f"{self.spec.observation_days}-day observation window."
            ),
            "spec": asdict(self.spec),
            "plan": self.plan.to_dict(),
            "sets": {
                "fit": prev(self.fit),
                "valid": prev(self.valid),
                "refit": prev(self.refit),
                "test": prev(self.test),
            },
            "per_cutoff_and_group": per_cutoff.to_dict(orient="records"),
        }


def build_churn_dataset(events: pd.DataFrame, cfg: dict[str, Any]) -> ChurnDataset:
    spec = ChurnSpec.from_config(cfg)
    if not events["timestamp"].is_monotonic_increasing:
        raise ValueError("events must be sorted by timestamp.")
    plan = plan_cutoffs(events["timestamp"].iloc[0], events["timestamp"].iloc[-1], spec)
    first_seen = compute_first_seen(events)
    groups = split_avatar_groups(first_seen.index.to_numpy(), spec)

    cutoffs = sorted(plan.train_cutoffs + plan.test_cutoffs)
    frames = [build_snapshot(events, c, spec, cfg, first_seen, plan.data_end) for c in cutoffs]
    snaps = pd.concat(frames, ignore_index=True)

    group = np.full(len(snaps), "train", dtype=object)
    group[np.isin(snaps["avatar"].to_numpy(), groups["valid"])] = "valid"
    group[np.isin(snaps["avatar"].to_numpy(), groups["test"])] = "test"
    snaps["group"] = pd.Categorical(group, categories=["train", "valid", "test"])

    valid_cuts = set(plan.train_cutoffs[: spec.n_valid_cutoffs])        # newest training cutoffs
    older = set(plan.train_cutoffs[spec.n_valid_cutoffs :])             # everything before them
    train_cut = set(plan.train_cutoffs)
    g, c = snaps["group"], snaps["cutoff"]
    fit = snaps[(g == "train") & c.isin(older)]
    valid = snaps[(g == "valid") & c.isin(valid_cuts)]
    refit = snaps[g.isin(["train", "valid"]) & c.isin(train_cut)]
    test = snaps[(g == "test") & c.isin(set(plan.test_cutoffs))]

    ds = ChurnDataset(spec, plan, snaps, fit.reset_index(drop=True), valid.reset_index(drop=True),
                      refit.reset_index(drop=True), test.reset_index(drop=True))
    verify_split_integrity(ds)
    return ds


def verify_split_integrity(ds: ChurnDataset) -> None:
    """Hard checks run on every dataset build (also covered by unit tests)."""
    fit_av, val_av, test_av = (set(d["avatar"]) for d in (ds.fit, ds.valid, ds.test))
    refit_av = set(ds.refit["avatar"])
    if refit_av & test_av or fit_av & test_av or val_av & test_av:
        raise LeakageError("An avatar appears in both training/validation and test data.")
    if fit_av & val_av:
        raise LeakageError("An avatar appears in both fit and validation data.")
    pred = pd.Timedelta(days=ds.spec.prediction_days)
    if len(ds.fit) and len(ds.valid) and (ds.fit["cutoff"].max() + pred) > ds.valid["cutoff"].min():
        raise LeakageError("A fitting label window overlaps the validation period.")
    obs = pd.Timedelta(days=ds.spec.observation_days)
    if len(ds.refit) and len(ds.test) and (ds.refit["cutoff"].max() + pred) > (ds.test["cutoff"].min() - obs):
        raise LeakageError("A training label window reaches into the test observation period.")
    if len(ds.test) and ds.test["cutoff"].min() <= ds.refit["cutoff"].max():
        raise LeakageError("Test cutoffs are not newer than every training cutoff.")
