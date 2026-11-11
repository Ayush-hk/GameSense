"""Tests for the Game Lab on tiny hand-built tables (never the real data)."""
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.errors import DataReadError, DatasetNotFoundError, SchemaError
from src.games import analysis as A
from src.games import models as M
from src.games.names import normalize
from src.games.openplay import assign_age_band, load_open_play
from src.games.steamcat import owners_midpoint

CFG = {"games": {
    "open_play_dir": "op", "steam_insights_dir": "si", "out_dir": "out", "processed_dir": "proc",
    "age_bands": [18, 23, 27, 99], "min_group_participants": 2,
    "frustration": {"items": ["bangs_2", "bangs_4", "bangs_6"], "satisfaction_items": ["bangs_1", "bangs_3", "bangs_5"],
                    "high_threshold": 3.0, "min_participants_per_game": 2, "min_entries_per_game": 2},
    "recommender": {"min_hours_liked": 5, "min_user_games": 3, "k": 3, "folds": 3},
    "random_state": 1}}


class NamesTests(unittest.TestCase):
    def test_variants_match(self):
        self.assertEqual(normalize("Overwatch® 2"), normalize("overwatch 2"))
        self.assertEqual(normalize("Baldur's Gate III"), normalize("BG3"))
        self.assertEqual(normalize("The Sims 4"), "sims 4")
        self.assertEqual(normalize(None), "")
        self.assertEqual(normalize(float("nan")), "")

    def test_ambiguous_not_merged(self):
        self.assertNotEqual(normalize("Zelda"), normalize("Mario Kart"))


class AnalysisTests(unittest.TestCase):
    def test_bh_fdr_monotone_and_bounded(self):
        p = np.array([0.001, 0.01, 0.03, 0.5])
        q = A.bh_fdr(p)
        self.assertTrue(np.all(q >= p - 1e-12))
        self.assertTrue(np.all(q <= 1))
        self.assertAlmostEqual(q[0], 0.004)

    def test_genre_harmonise(self):
        self.assertEqual(A.harmonise_genres("Role Playing, Role-playing (RPG), Other, Platform"), ["RPG", "Platformer"])
        self.assertEqual(A.harmonise_genres(None), [])

    def test_cronbach_perfect_and_independent(self):
        x = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "c": [1, 2, 3, 4, 5]})
        self.assertAlmostEqual(A.cronbach_alpha(x), 1.0)
        rng = np.random.default_rng(0)
        y = pd.DataFrame(rng.normal(size=(500, 3)))
        self.assertLess(abs(A.cronbach_alpha(y)), 0.2)

    def _diary(self):
        rows = []
        for pid in range(10):
            for g, base in (("hard", 4.0), ("easy", 1.5)):
                for _ in range(3):
                    f = base + (0.1 if pid % 2 else -0.1)
                    rows.append({"pid": f"p{pid}", "game_key": g, "game_label": g, "frustration": f,
                                 "high_frustration": int(f >= 3), "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=len(rows))})
        return pd.DataFrame(rows)

    def test_shrunken_ranks_and_min_rule(self):
        d = self._diary()
        g = A.shrunken_frustration(d, min_participants=5, min_entries=10)
        self.assertEqual(list(g["game_key"]), ["hard", "easy"])
        self.assertTrue((g["shrunken_mean"].iloc[0] < g["mean"].iloc[0]))  # pulled toward grand mean
        self.assertTrue(A.shrunken_frustration(d, min_participants=11, min_entries=10).empty)

    def test_within_person_delta_sign(self):
        g = A.within_person_delta(self._diary(), min_participants=5, min_entries=10)
        self.assertGreater(g.set_index("game_key").loc["hard", "delta"], 0)
        self.assertLess(g.set_index("game_key").loc["easy", "delta"], 0)

    def _playtime(self):
        rows = []
        for pid, band in (("a", "18-22"), ("b", "18-22"), ("c", "36+"), ("d", "36+")):
            for game, genre, h in (("G1", "Shooter", 10), ("G2", "Family + Kids", 5)):
                hours = h * (3 if (band == "18-22") == (genre == "Shooter") else 1)
                rows.append({"pid": pid, "game_key": game, "game_name": game, "genres_clean": genre, "hours": hours,
                             "platform": "Steam", "named": True})
        return pd.DataFrame(rows), pd.DataFrame({"pid": list("abcd"), "age_band": ["18-22", "18-22", "36+", "36+"], "age": [19, 20, 40, 41]})

    def test_age_genre_lift_direction_and_privacy(self):
        pt, people = self._playtime()
        m = A.age_genre_lift(A.age_genre_matrix(pt, people, min_participants=2))
        x = m.set_index(["age_band", "genre"])
        self.assertGreater(x.loc[("18-22", "Shooter"), "lift"], 1)
        self.assertGreater(x.loc[("36+", "Family"), "lift"], 1)
        self.assertTrue(A.age_genre_matrix(pt, people, min_participants=3).empty)

    def test_most_played_min_players(self):
        pt, _ = self._playtime()
        self.assertEqual(len(A.most_played(pt, min_players=4)), 2)
        self.assertEqual(len(A.most_played(pt, min_players=5)), 0)


class ModelTests(unittest.TestCase):
    def test_cluster_bootstrap_ci_contains_auc(self):
        rng = np.random.default_rng(0)
        g = np.repeat(np.arange(60), 5)
        y = rng.integers(0, 2, len(g))
        p = y * 0.5 + rng.random(len(g))
        lo, hi = M.cluster_bootstrap_auc(y, p, g, n_boot=100)
        self.assertLessEqual(lo, M._safe_auc(y, p))
        self.assertGreaterEqual(hi, M._safe_auc(y, p))

    def test_person_history_uses_only_earlier_rows(self):
        d = pd.DataFrame({"pid": ["a"] * 3, "date": pd.date_range("2025-01-01", periods=3), "frustration": [1.0, 3.0, 7.0]})
        h = M._person_history(d)
        self.assertTrue(np.isnan(h.iloc[0]))
        self.assertAlmostEqual(h.iloc[1], 1.0)
        self.assertAlmostEqual(h.iloc[2], 2.0)

    def test_loo_game_prior_excludes_own_person(self):
        d = pd.DataFrame({"pid": ["a", "b", "c"], "game_key": ["g"] * 3, "frustration": [1.0, 5.0, 5.0]})
        p = M._loo_game_prior(d, grand=3.0, m=0.0001)
        self.assertAlmostEqual(p.iloc[0], 5.0, places=2)
        self.assertAlmostEqual(p.iloc[1], 3.0, places=2)

    def test_similarity_and_recommend(self):
        # items 0 and 1 always together, item 2 separate
        M_ = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
        sim = M.item_similarity(M_, shrink=0.1)
        self.assertEqual(sim.diagonal().sum(), 0)
        self.assertGreater(sim[0, 1], sim[0, 2])
        s = M.recommend_scores(sim, [0])
        self.assertEqual(int(np.argmax(s)), 1)
        self.assertEqual(s[0], -np.inf)

    def test_recommender_beats_random_on_structured_data(self):
        rng = np.random.default_rng(0)
        blocks = [list(range(0, 10)), list(range(10, 20))]
        rows = []
        for u in range(120):
            b = blocks[u % 2]
            rows.append(np.isin(np.arange(20), rng.choice(b, 6, replace=False)).astype(np.float32))
        ev = M.evaluate_recommender(np.array(rows), CFG)["metrics"].set_index(["model", "k"])["hit_rate"]
        self.assertGreater(ev[("item_item", 3)], ev[("random", 3)] + 0.2)


class LoaderTests(unittest.TestCase):
    def _write(self, d: Path, name: str, df: pd.DataFrame):
        with gzip.open(d / name, "wt") as f:
            df.to_csv(f, index=False)

    def test_missing_file_message(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(DatasetNotFoundError):
                load_open_play(CFG, root=t)

    def test_bad_schema_and_good_load(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "op"
            d.mkdir()
            self._write(d, "survey_intake.csv.gz", pd.DataFrame({"pid": ["a", "b", "c"], "qualified": [True, True, False],
                       "age": [19, 30, 25], "gender": ["Man", "Woman", "Man"], "country": ["US", "UK", "US"]}))
            self._write(d, "survey_daily.csv.gz", pd.DataFrame({"pid": ["a", "a", "b"], "date": ["2025-01-01", "2025-01-02", "2025-01-01"],
                       "played24hr": ["Yes", "Yes", "Yes"], **{f"bangs_{i}": [4, 2, 1] for i in range(1, 7)},
                       "most_recent_game": ["Overwatch® 2", "Overwatch 2", "Celeste"], "most_recent_game_plat": ["Steam", "Steam", "Steam"]}))
            self._write(d, "steam.csv.gz", pd.DataFrame({"pid": ["a", "b", "c"], "title_id": ["x", "y", "x"], "minutes": [120, 60, 30], "steam_id": [1, 2, 3]}))
            self._write(d, "nintendo.csv.gz", pd.DataFrame({"pid": ["a"], "title_id": ["n1"], "duration": [60]}))
            self._write(d, "xbox.csv.gz", pd.DataFrame({"pid": ["b"], "title_id": ["xb1"], "duration": [30]}))
            meta = pd.DataFrame({"original_name": ["x", "y", "n1", "xb1"], "platform": ["Steam", "Steam", "Nintendo", "Xbox"],
                                 "igdb_name": ["Overwatch 2", "Celeste", None, None], "match_confidence": [1.0, 0.9, None, None],
                                 "genres": ["Shooter", "Platform", None, None], "game_modes": ["Multiplayer", "Single player", None, None],
                                 "steam_pct_positive": [0.5, 0.97, None, None], "xbox_primary_genre": [None, None, None, "Racing + Flying"]})
            self._write(d, "game_metadata.csv.gz", meta)
            op = load_open_play(CFG, root=t)
            self.assertEqual(len(op.participants), 2)                        # unqualified removed
            self.assertEqual(len(op.diary), 3)
            self.assertEqual(op.diary["game_key"].nunique(), 2)              # Overwatch variants merged
            self.assertAlmostEqual(op.diary["frustration"].iloc[0], 4.0)     # items 2,4,6 all rated 4
            self.assertEqual(int(op.diary["high_frustration"].iloc[0]), 1)
            self.assertEqual(int(op.diary["high_frustration"].iloc[1]), 0)
            self.assertTrue(op.playtime.loc[op.playtime["platform"] == "Xbox", "genres_clean"].iloc[0] == "Racing + Flying")
            self.assertFalse(op.playtime.loc[op.playtime["platform"] == "Xbox", "named"].iloc[0])
            # corrupt schema
            self._write(d, "survey_intake.csv.gz", pd.DataFrame({"pid": ["a"]}))
            with self.assertRaises(SchemaError):
                load_open_play(CFG, root=t)

    def test_age_band_labels(self):
        b = assign_age_band(pd.Series([18, 22, 23, 50]), [18, 23, 27, 99])
        self.assertEqual(list(b), ["18-22", "18-22", "23-26", "27+"])

    def test_owner_midpoint(self):
        self.assertEqual(owners_midpoint(pd.Series(["10,000,000 .. 20,000,000"])).iloc[0], 15_000_000)


class RegressionTests(unittest.TestCase):
    """Bugs found while testing the finished project; each one failed before the fix."""

    def test_harmonise_genres_accepts_nan_and_none(self):
        # page 6 crashed with "'float' object has no attribute 'split'" when a genre was chosen
        self.assertEqual(A.harmonise_genres(float("nan")), [])
        self.assertEqual(A.harmonise_genres(None), [])
        self.assertEqual(A.harmonise_genres(""), [])

    def test_empty_shrunken_frame_keeps_columns(self):
        # page 8 crashed with KeyError 'shrunken_mean' when no game met the minimum
        d = pd.DataFrame({"pid": ["a", "b"], "game_key": ["g", "g"], "frustration": [2.0, 3.0], "high_frustration": [0, 1]})
        g = A.shrunken_frustration(d, min_participants=10, min_entries=1)
        self.assertTrue(g.empty)
        for c in ("shrunken_mean", "ci_low", "ci_high", "grand_mean"):
            self.assertIn(c, g.columns)

    def test_frustration_by_genre_with_no_matching_games(self):
        d = pd.DataFrame({"pid": ["a"], "game_key": ["g"], "frustration": [2.0], "high_frustration": [0]})
        games = pd.DataFrame({"game_key": ["other"], "genres": ["Shooter"]})
        self.assertTrue(A.frustration_by_genre(d, games, 1, 1).empty)

    def test_privacy_minimum_is_consistent_in_real_config(self):
        from src.config import load_config
        gc = load_config()["games"]
        self.assertGreaterEqual(gc["frustration"]["min_participants_per_game"], gc["min_group_participants"])

    def test_unjoined_steam_file_message_names_the_fix(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "op"
            d.mkdir()
            (d / "survey_intake.csv.gz").write_bytes(b"")
            (d / "steam.csv.gz.part1").write_bytes(b"x")
            from src.games.openplay import _read
            with self.assertRaises(DatasetNotFoundError) as cm:
                _read(d / "steam.csv.gz", ["pid"], "steam")
            self.assertIn("join_parts", str(cm.exception))

    def test_damaged_gz_gives_friendly_error(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            with gzip.open(d / "x.csv.gz", "wt") as f:
                f.write("pid,a\n" + "p1,1\n" * 5000)
            raw = (d / "x.csv.gz").read_bytes()
            (d / "x.csv.gz").write_bytes(raw[: len(raw) // 2])
            from src.games.openplay import _read
            with self.assertRaises(DataReadError):
                _read(d / "x.csv.gz", ["pid"], "x")

    def test_damaged_steam_zip_gives_friendly_error(self):
        with tempfile.TemporaryDirectory() as t:
            si = Path(t) / "si"
            si.mkdir()
            (si / "games.zip").write_bytes(b"this is not a zip")
            from src.games.steamcat import load_catalogue
            with self.assertRaises(DataReadError):
                load_catalogue(CFG, root=t)


if __name__ == "__main__":
    unittest.main()
