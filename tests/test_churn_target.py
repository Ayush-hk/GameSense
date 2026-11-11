import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from src.config import load_config
from src.errors import InsufficientHistoryError, LeakageError
from src.features.behavior import compute_first_seen
from src.features.churn_target import (
    ChurnSpec,
    assert_no_leakage,
    build_churn_dataset,
    build_snapshot,
    plan_cutoffs,
    split_avatar_groups,
    verify_split_integrity,
)
from tests.fixtures import make_events_frame


def _events(**kw):
    df = make_events_frame(**kw).rename(columns={"char": "avatar"})
    for c in ("race", "charclass", "zone"):
        df[c] = df[c].astype("category")
    return df


class ChurnTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.ev = _events(n_avatars=300, days=330, seed=5)
        cls.ds = build_churn_dataset(cls.ev, cls.cfg)

    def test_label_matches_manual_definition(self):
        spec = ChurnSpec.from_config(self.cfg)
        cutoff = self.ds.plan.test_cutoff
        snap = build_snapshot(self.ev, cutoff, spec, self.cfg, compute_first_seen(self.ev), self.ds.plan.data_end)
        obs = self.ev[(self.ev.timestamp >= cutoff - pd.Timedelta(days=28)) & (self.ev.timestamp < cutoff)]
        fut = self.ev[(self.ev.timestamp >= cutoff) & (self.ev.timestamp < cutoff + pd.Timedelta(days=28))]
        self.assertEqual(set(snap.avatar), set(obs.avatar))
        expected = {a: int(a not in set(fut.avatar)) for a in set(obs.avatar)}
        self.assertEqual(dict(zip(snap.avatar, snap.churned)), expected)

    def test_only_active_avatars_scored(self):
        self.assertTrue((self.ds.snapshots["active_days"] >= 1).all())

    def test_test_cutoff_newest_and_followup_complete(self):
        plan, spec = self.ds.plan, self.ds.spec
        self.assertTrue(all(c < plan.test_cutoff for c in plan.train_cutoffs))
        self.assertLessEqual(plan.test_cutoff + pd.Timedelta(days=spec.prediction_days), plan.data_end)

    def test_avatar_disjoint_and_out_of_time(self):
        verify_split_integrity(self.ds)
        self.assertFalse(set(self.ds.refit.avatar) & set(self.ds.test.avatar))
        self.assertGreater(self.ds.test.cutoff.min(), self.ds.refit.cutoff.max())

    def test_all_sets_have_both_classes(self):
        for name in ("fit", "valid", "refit", "test"):
            self.assertEqual(getattr(self.ds, name)["churned"].nunique(), 2, name)

    def test_label_independent_of_features_window_future(self):
        """Features of a snapshot must not change if events after the cutoff are removed."""
        spec = ChurnSpec.from_config(self.cfg)
        cutoff = self.ds.plan.train_cutoffs[0]
        full = build_snapshot(self.ev, cutoff, spec, self.cfg, compute_first_seen(self.ev))
        trimmed_ev = self.ev[self.ev.timestamp < cutoff].reset_index(drop=True)
        trimmed = build_snapshot(trimmed_ev, cutoff, spec, self.cfg, compute_first_seen(trimmed_ev))
        cols = [c for c in full.columns if c not in ("churned", "observed_tenure_days")]
        a = full[cols].sort_values("avatar").reset_index(drop=True)
        b = trimmed[cols].sort_values("avatar").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)

    def test_assert_no_leakage(self):
        assert_no_leakage(pd.Timestamp("2008-02-01"), pd.Timestamp("2008-02-01"))
        with self.assertRaises(LeakageError):
            assert_no_leakage(pd.Timestamp("2008-02-02"), pd.Timestamp("2008-02-01"))

    def test_verify_detects_shared_avatar(self):
        bad = replace(self.ds, test=pd.concat([self.ds.test, self.ds.refit.head(3)], ignore_index=True))
        with self.assertRaises(LeakageError):
            verify_split_integrity(bad)

    def test_insufficient_followup_rejected(self):
        spec = ChurnSpec.from_config(self.cfg)
        late = self.ds.plan.data_end - pd.Timedelta(days=5)
        with self.assertRaises(InsufficientHistoryError):
            build_snapshot(self.ev, late, spec, self.cfg, compute_first_seen(self.ev), self.ds.plan.data_end)

    def test_short_history_rejected(self):
        short = _events(n_avatars=20, days=60, seed=1)
        with self.assertRaises(InsufficientHistoryError):
            build_churn_dataset(short, self.cfg)

    def test_split_deterministic_and_disjoint(self):
        spec = ChurnSpec.from_config(self.cfg)
        ids = np.arange(1, 501)
        g1, g2 = split_avatar_groups(ids, spec), split_avatar_groups(ids, spec)
        for k in g1:
            np.testing.assert_array_equal(g1[k], g2[k])
        self.assertEqual(sum(len(v) for v in g1.values()), 500)
        self.assertFalse(set(g1["train"]) & set(g1["test"]))
        self.assertFalse(set(g1["valid"]) & set(g1["test"]))

    def test_plan_cutoffs_spacing(self):
        spec = ChurnSpec.from_config(self.cfg)
        plan = plan_cutoffs(pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31 12:00"), spec)
        self.assertEqual(plan.data_end, pd.Timestamp("2008-12-31"))
        self.assertEqual(plan.test_cutoff, pd.Timestamp("2008-12-03"))
        self.assertEqual(plan.test_cutoffs[-1], plan.test_cutoff - pd.Timedelta(days=56))
        # newest training label window ends exactly where the earliest test observation window begins
        self.assertEqual(plan.train_cutoffs[0] + pd.Timedelta(days=28), plan.test_cutoffs[-1] - pd.Timedelta(days=28))


if __name__ == "__main__":
    unittest.main()
