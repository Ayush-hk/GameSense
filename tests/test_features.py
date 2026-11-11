import unittest

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.behavior import FEATURE_COLUMNS, build_window_features, compute_first_seen
from src.features.sessions import sessionize
from tests.fixtures import tiny_frame


class SessionTests(unittest.TestCase):
    def test_gap_splits_sessions_and_duration_includes_poll(self):
        s = sessionize(tiny_frame(), gap_minutes=30, poll_minutes=10).sort_values(["avatar", "start"]).reset_index(drop=True)
        a1 = s[s["avatar"] == 1]
        self.assertEqual(len(a1), 2)
        self.assertEqual(a1["n_polls"].tolist(), [4, 2])
        # 4 polls 10:00..10:30 -> 30 min span + 10 min poll interval
        self.assertEqual(a1["duration_min"].tolist(), [40.0, 20.0])
        a2 = s[s["avatar"] == 2]
        self.assertEqual(len(a2), 2)                     # 60-minute gap > 30 -> two sessions
        self.assertEqual(a2["duration_min"].tolist(), [10.0, 10.0])

    def test_gap_equal_to_limit_stays_one_session(self):
        t = pd.Timestamp("2008-01-01")
        df = pd.DataFrame({"avatar": [1, 1], "timestamp": [t, t + pd.Timedelta(minutes=30)]})
        self.assertEqual(len(sessionize(df, 30, 10)), 1)

    def test_empty_input(self):
        out = sessionize(tiny_frame().iloc[0:0], 30, 10)
        self.assertEqual(len(out), 0)
        self.assertIn("duration_min", out.columns)


class WindowFeatureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.ev = tiny_frame()

    def test_known_values(self):
        start, end = pd.Timestamp("2008-01-01"), pd.Timestamp("2008-01-08")
        f = build_window_features(self.ev, start, end, self.cfg, compute_first_seen(self.ev)).set_index("avatar")
        a1, a2 = f.loc[1], f.loc[2]
        self.assertEqual(list(f.columns), [c for c in FEATURE_COLUMNS if c != "avatar"])
        self.assertEqual(a1["active_days"], 2)
        self.assertEqual(a1["n_polls"], 6)
        self.assertAlmostEqual(a1["est_hours_online"], 6 * 10 / 60)
        self.assertEqual(a1["n_sessions"], 2)
        self.assertAlmostEqual(a1["mean_session_min"], 30.0)      # (40 + 20) / 2
        self.assertEqual(a1["max_session_min"], 40.0)
        self.assertEqual(a1["zones_visited"], 2)
        self.assertEqual(a1["has_guild"], 1)
        self.assertEqual(a2["has_guild"], 0)
        self.assertEqual(a1["level_gain"], 1)                     # level 10 -> 11
        self.assertEqual(a2["n_sessions"], 2)
        self.assertEqual(a2["active_days"], 1)
        self.assertEqual(a1["race"], "Orc")
        # last activity 2008-01-02 10:10 -> window end 2008-01-08 00:00
        self.assertAlmostEqual(a1["days_since_last_active"], (end - pd.Timestamp("2008-01-02 10:10")) / pd.Timedelta(days=1))

    def test_window_excludes_future_rows(self):
        """Rows at or after the window end must not influence any feature (leakage guard)."""
        start, end = pd.Timestamp("2008-01-01"), pd.Timestamp("2008-01-02")   # excludes day 2
        f = build_window_features(self.ev, start, end, self.cfg).set_index("avatar")
        self.assertEqual(f.loc[1, "n_polls"], 4)
        self.assertEqual(f.loc[1, "level_last"], 10)
        self.assertEqual(f.loc[1, "zones_visited"], 1)

    def test_features_do_not_change_when_future_data_is_added(self):
        start, end = pd.Timestamp("2008-01-01"), pd.Timestamp("2008-01-02")
        before = build_window_features(self.ev, start, end, self.cfg)
        future = pd.DataFrame(
            {"avatar": [1, 3], "level": [60, 40], "race": ["Orc", "Orc"], "charclass": ["Mage", "Mage"],
             "zone": ["Z", "Z"], "guild": [9, 9], "timestamp": [pd.Timestamp("2008-03-01"), pd.Timestamp("2008-03-02")]}
        )
        for c in ("race", "charclass", "zone"):
            future[c] = future[c].astype("category")
        bigger = pd.concat([self.ev, future], ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
        after = build_window_features(bigger, start, end, self.cfg)
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True), check_dtype=False)

    def test_empty_window_returns_empty_frame_with_schema(self):
        f = build_window_features(self.ev, pd.Timestamp("2010-01-01"), pd.Timestamp("2010-01-08"), self.cfg)
        self.assertEqual(len(f), 0)
        self.assertEqual(list(f.columns), FEATURE_COLUMNS)

    def test_bad_window_raises(self):
        with self.assertRaises(ValueError):
            build_window_features(self.ev, pd.Timestamp("2008-01-08"), pd.Timestamp("2008-01-01"), self.cfg)

    def test_features_are_finite_and_in_range(self):
        from tests.fixtures import make_events_frame
        df = make_events_frame(n_avatars=40, days=60, seed=3).rename(columns={"char": "avatar"})
        for c in ("race", "charclass", "zone"):
            df[c] = df[c].astype("category")
        f = build_window_features(df, df["timestamp"].min().floor("D"), df["timestamp"].max().ceil("D"), self.cfg)
        num = f.drop(columns=["race", "charclass", "avatar", "observed_tenure_days"])
        self.assertTrue(np.isfinite(num.to_numpy(dtype=float)).all())
        for c in ("weekend_share", "night_share", "active_day_ratio"):
            self.assertTrue(f[c].between(0, 1).all(), c)
        self.assertTrue((f["active_days"] >= 1).all())
        self.assertTrue((f["activity_regularity"] <= 1.0 + 1e-9).all())


if __name__ == "__main__":
    unittest.main()
