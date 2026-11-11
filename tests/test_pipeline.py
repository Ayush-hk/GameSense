"""End-to-end pipeline on a SYNTHETIC TEST FIXTURE in a temporary project root.

This verifies the code path only. Nothing it produces is a result on the Kaggle dataset.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import get_paths, load_config
from src.errors import ArtifactMissingError, DatasetNotFoundError
from src.explainability.explain import shap_available
from src.pipeline import run_pipeline, stage_explain
from src.reporting.report import build_insights, write_evaluation_report
from src.utils import read_json
from tests.fixtures import make_events_frame, write_events_csv


class PipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.cfg["models"]["random_forest"]["n_estimators"] = 50
        cls.cfg["explain"]["permutation_repeats"] = 2
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.csv = write_events_csv(cls.root / "in" / "wowah_data.csv", make_events_frame(n_avatars=300, days=330, seed=11))
        cls.status = run_pipeline(cls.cfg, raw_file=cls.csv, root=cls.root, data_origin="synthetic_test_fixture")
        cls.paths = get_paths(cls.cfg, root=cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_stages_ok(self):
        for stage, s in self.status.items():
            self.assertEqual(s["status"], "ok", (stage, s))

    def test_outputs_exist(self):
        p = self.paths
        for f in (p.events_table, p.profiles_table, p.churn_snapshots, p.daily_activity, p.hourly_activity, p.dataset_profile,
                  p.dataset_card, p.segmentation_report, p.cluster_profiles, p.metrics, p.churn_model, p.seg_preprocessor,
                  p.kmeans_model, p.permutation_importance, p.explain_report, p.evaluation_report, p.insights):
            self.assertTrue(f.is_file(), f)

    def test_synthetic_origin_is_flagged_in_reports(self):
        self.assertEqual(read_json(self.paths.run_info)["data_origin"], "synthetic_test_fixture")
        self.assertIn("NOT from the Kaggle dataset", self.paths.evaluation_report.read_text(encoding="utf-8"))

    def test_json_files_strict_and_consistent(self):
        prof = read_json(self.paths.dataset_profile)
        events = pd.read_pickle(self.paths.events_table)
        self.assertEqual(prof["rows"], len(events))
        self.assertEqual(prof["avatars"], events["avatar"].nunique())

    def test_explain_fallback_or_shap(self):
        rep = read_json(self.paths.explain_report)
        if shap_available():
            self.assertEqual(rep["method"], "SHAP")
        else:
            self.assertIn("permutation", rep["method"])
            self.assertIn("not installed", rep["note"])
        self.assertTrue(rep["top_features"])
        self.assertIn("not proof", rep["caveat"].lower() + " not proof")

    def test_insights_come_from_files(self):
        titles = [i["title"] for i in build_insights(self.paths)]
        self.assertIn("Churn prediction performance", titles)
        self.assertIn("Player segments", titles)

    def test_downstream_stage_without_ingest_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ArtifactMissingError):
                run_pipeline(self.cfg, stages=["segment"], root=d)
            with self.assertRaises(ArtifactMissingError):
                stage_explain(self.cfg, get_paths(self.cfg, root=d))
            self.assertEqual(read_json(get_paths(self.cfg, root=d).pipeline_status)["segment"]["status"], "failed")

    def test_missing_raw_file_gives_instructions(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(DatasetNotFoundError) as ctx:
                run_pipeline(self.cfg, stages=["ingest"], root=d)
            self.assertIn("kaggle.com", str(ctx.exception))

    def test_unknown_stage(self):
        with self.assertRaises(ValueError):
            run_pipeline(self.cfg, stages=["nope"], root=self.root)

    def test_report_without_results_says_not_available(self):
        with tempfile.TemporaryDirectory() as d:
            paths = get_paths(self.cfg, root=d).ensure()
            text = write_evaluation_report(paths, self.cfg)
            self.assertIn("Not available", text)
            self.assertNotIn("ROC-AUC 0.", text)


if __name__ == "__main__":
    unittest.main()
