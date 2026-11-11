import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import get_paths, load_config
from src.data.loader import (
    detect_timestamp_format,
    find_raw_file,
    load_events,
    load_raw_events,
    resolve_columns,
    save_events,
)
from src.errors import (
    ArtifactMissingError,
    DatasetNotFoundError,
    EmptyDatasetError,
    SchemaError,
    TimestampParseError,
)
from tests.fixtures import make_events_frame, write_events_csv


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.df = make_events_frame(n_avatars=30, days=40, seed=1)
        cls.csv = write_events_csv(cls.dir / "wowah_data.csv", cls.df)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loads_padded_headers_and_us_timestamps(self):
        events, report = load_raw_events(self.csv, self.cfg)
        self.assertEqual(report["rows_read"], len(self.df))
        self.assertEqual(report["timestamp_format"], "%m/%d/%y %H:%M:%S")
        self.assertEqual(len(events), len(self.df) - report["duplicate_avatar_timestamp_rows_removed"])
        self.assertTrue(events["timestamp"].is_monotonic_increasing)
        self.assertEqual(str(events["timestamp"].dtype), "datetime64[ns]")
        self.assertEqual(set(events.columns), {"avatar", "level", "race", "charclass", "zone", "guild", "timestamp"})
        self.assertEqual(events["avatar"].nunique(), self.df["char"].nunique())

    def test_iso_timestamps_also_work(self):
        p = write_events_csv(self.dir / "iso.csv", self.df, fmt="%Y-%m-%d %H:%M:%S", pad_header=False)
        events, report = load_raw_events(p, self.cfg)
        self.assertEqual(report["timestamp_format"], "%Y-%m-%d %H:%M:%S")
        self.assertEqual(events["timestamp"].min(), self.df["timestamp"].min().floor("s"))

    def test_missing_column_raises_schema_error(self):
        bad = pd.read_csv(self.csv, skipinitialspace=True).drop(columns=["level"])
        p = self.dir / "nolevel.csv"
        bad.to_csv(p, index=False)
        with self.assertRaises(SchemaError) as ctx:
            load_raw_events(p, self.cfg)
        self.assertIn("level", str(ctx.exception))

    def test_optional_columns_may_be_absent(self):
        """The lightweight WoWAH subset has only player_id, level, datetime."""
        raw = pd.read_csv(self.csv, skipinitialspace=True)[["char", "level", "timestamp"]]
        raw = raw.rename(columns={"char": "player_id", "timestamp": "datetime"})
        p = self.dir / "lite.csv"
        raw.to_csv(p, index=False)
        events, report = load_raw_events(p, self.cfg)
        self.assertEqual(set(events.columns), {"avatar", "level", "timestamp"})
        self.assertEqual(report["optional_columns_absent"], ["race", "charclass", "zone", "guild"])

    def test_unparseable_timestamps_fail_loudly(self):
        raw = pd.read_csv(self.csv, skipinitialspace=True)
        raw["timestamp"] = "not-a-date"
        p = self.dir / "baddates.csv"
        raw.to_csv(p, index=False)
        with self.assertRaises(TimestampParseError):
            load_raw_events(p, self.cfg)

    def test_invalid_rows_are_dropped_and_counted(self):
        raw = pd.read_csv(self.csv, skipinitialspace=True)
        raw.loc[0, "level"] = 999          # out of range
        raw.loc[1, "char"] = None          # unusable avatar
        raw.loc[2, "zone"] = None          # becomes Unknown
        raw.loc[3, "guild"] = None         # becomes "no guild"
        p = self.dir / "dirty.csv"
        raw.to_csv(p, index=False)
        events, report = load_raw_events(p, self.cfg)
        self.assertEqual(report["dropped_invalid_level"], 1)
        self.assertEqual(report["dropped_unusable_avatar_or_timestamp"], 1)
        self.assertEqual(report["missing_category_labelled_unknown"]["zone"], 1)
        self.assertEqual(report["missing_guild_treated_as_no_guild"], 1)
        self.assertIn("Unknown", set(events["zone"].astype(str)))

    def test_duplicates_removed(self):
        raw = pd.read_csv(self.csv, skipinitialspace=True)
        raw = pd.concat([raw, raw.head(25)], ignore_index=True)
        p = self.dir / "dups.csv"
        raw.to_csv(p, index=False)
        _, report = load_raw_events(p, self.cfg)
        self.assertGreaterEqual(report["duplicate_avatar_timestamp_rows_removed"], 25)

    def test_empty_file(self):
        p = self.dir / "empty.csv"
        p.write_text("char,level,race,charclass,zone,guild,timestamp\n")
        with self.assertRaises(EmptyDatasetError):
            load_raw_events(p, self.cfg)

    def test_missing_file(self):
        with self.assertRaises(DatasetNotFoundError):
            load_raw_events(self.dir / "nope.csv", self.cfg)

    def test_find_raw_file_instructions_when_absent(self):
        paths = get_paths(self.cfg, root=self.dir / "emptyproj")
        with self.assertRaises(DatasetNotFoundError) as ctx:
            find_raw_file(self.cfg, paths)
        msg = str(ctx.exception)
        self.assertIn("kaggle.com/datasets/mylesoneill/warcraft-avatar-history", msg)
        self.assertIn("wowah_data.csv", msg)

    def test_find_raw_file_uses_single_csv_fallback(self):
        paths = get_paths(self.cfg, root=self.dir / "proj2")
        paths.raw_dir.mkdir(parents=True)
        (paths.raw_dir / "renamed.csv").write_text("x")
        self.assertEqual(find_raw_file(self.cfg, paths).name, "renamed.csv")

    def test_resolve_columns_and_detect_format(self):
        m = resolve_columns([" char", " Level", "race ", "CharClass", "zone", "guild", "timestamp"], self.cfg["dataset"]["columns"])
        self.assertEqual(m["avatar"], " char")
        fmt, rate = detect_timestamp_format(pd.Series(["01/31/08 10:00:00"] * 5), self.cfg["dataset"]["timestamp_formats"], 0.99)
        self.assertEqual((fmt, rate), ("%m/%d/%y %H:%M:%S", 1.0))

    def test_processed_events_roundtrip_and_missing_artifact(self):
        events, _ = load_raw_events(self.csv, self.cfg)
        p = self.dir / "proc" / "events.pkl"
        save_events(events, p)
        back = load_events(p)
        pd.testing.assert_frame_equal(events.reset_index(drop=True), back.reset_index(drop=True))
        with self.assertRaises(ArtifactMissingError):
            load_events(self.dir / "does_not_exist.pkl")


if __name__ == "__main__":
    unittest.main()
