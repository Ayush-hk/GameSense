"""Version-independent copies of saved tables and the recommender."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.games import models as M
from src.portable import csv_path, read_table, save_table, table_exists


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "pid": ["a", "b", None, ""],
        "n": pd.Series([1, 2, 3, 4], dtype="int8"),
        "x": [0.1, float("nan"), 1e-9, 12345.678901234567],
        "flag": [True, False, True, True],
        "when": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-03", "2026-03-04"], utc=True),
        "grp": pd.Categorical(["train", "valid", "test", "train"], categories=["train", "valid", "test"]),
    })


class PortableTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Path(self.tmp.name) / "t.pkl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_copy_round_trips_values_types_and_missing(self):
        df = sample()
        save_table(df, self.p)
        self.assertTrue(csv_path(self.p).is_file())
        self.p.unlink()                                   # only the portable copy remains
        self.assertTrue(table_exists(self.p))
        back = read_table(self.p)
        pd.testing.assert_frame_equal(df, back, check_dtype=False)
        self.assertEqual(str(back["grp"].dtype), "category")
        self.assertTrue(pd.isna(back.loc[2, "pid"]))
        self.assertEqual(back.loc[3, "pid"], "")          # empty text stays empty text, not missing

    def test_unreadable_pickle_falls_back_to_csv_copy(self):
        df = sample()
        save_table(df, self.p)
        self.p.write_bytes(b"not a pickle written by this pandas")
        pd.testing.assert_frame_equal(df, read_table(self.p), check_dtype=False)

    def test_unreadable_pickle_without_copy_raises(self):
        self.p.write_bytes(b"garbage")
        with self.assertRaises(Exception):
            read_table(self.p)


class PortableRecommenderTests(unittest.TestCase):
    def test_round_trip_keeps_recommendations(self):
        rng = np.random.default_rng(0)
        sim = rng.random((5, 5)).astype("float32")
        rec = {"sim": sim, "items": np.array(["g1", "g2", "g3", "g4", "g5"]), "names": np.array(["G1", "G2", "G3", "G4", "G5"], dtype=object),
               "popularity": np.arange(5, dtype="float32"),
               "evaluation": {"metrics": pd.DataFrame({"model": ["popularity"], "k": [10], "hit_rate": [0.2]}), "n_users": 7, "n_games": 5, "density": 0.3}}
        with tempfile.TemporaryDirectory() as d:
            M.save_recommender_portable(rec, Path(d))
            back = M.load_recommender(Path(d))
        pd.testing.assert_frame_equal(M.recommend(rec, ["g1"], 3), M.recommend(back, ["g1"], 3))
        self.assertEqual(back["evaluation"]["n_users"], 7)
        self.assertEqual(float(back["evaluation"]["metrics"]["hit_rate"].iloc[0]), 0.2)


if __name__ == "__main__":
    unittest.main()
