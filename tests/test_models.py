import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analytics.segmentation import assign_segments, fit_segmentation, make_preprocessor
from src.config import get_paths, load_config
from src.errors import ArtifactMissingError, EmptyDatasetError, PredictionInputError
from src.features.behavior import build_window_features
from src.features.churn_target import build_churn_dataset
from src.models.churn import best_f1_threshold, binary_metrics, train_and_evaluate
from src.models.predict import load_bundle, predict_churn, validate_features
from tests.fixtures import make_events_frame


def _events(**kw):
    df = make_events_frame(**kw).rename(columns={"char": "avatar"})
    for c in ("race", "charclass", "zone"):
        df[c] = df[c].astype("category")
    return df


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        ev = _events(n_avatars=200, days=150, seed=2)
        cls.profiles = build_window_features(ev, ev.timestamp.min().floor("D"), ev.timestamp.max().floor("D") + pd.Timedelta(days=1), cls.cfg).drop(columns=["observed_tenure_days"])
        cls.res = fit_segmentation(cls.profiles, cls.cfg)

    def test_report_consistent(self):
        r = self.res["report"]
        lo, hi = self.cfg["segmentation"]["k_range"]
        self.assertTrue(lo <= r["chosen_k"] <= hi)
        self.assertEqual(sum(r["cluster_sizes"].values()), r["avatars_clustered"])
        self.assertTrue(-1 <= r["silhouette_final_sample"] <= 1)
        self.assertEqual(len(self.res["selection"]), hi - lo + 1)

    def test_every_avatar_assigned_or_marked_unclustered(self):
        prof = self.res["avatar_profiles"]
        self.assertEqual(len(prof), len(self.profiles))
        few = prof["active_days"] < self.cfg["segmentation"]["min_active_days"]
        self.assertTrue((prof.loc[few, "segment"] == -1).all())
        self.assertTrue((prof.loc[~few, "segment"] >= 0).all())

    def test_descriptions_have_no_marketing_labels(self):
        text = " ".join(self.res["report"]["descriptors"].values()).lower()
        for word in ("whale", "loyal", "at-risk", "casual"):
            self.assertNotIn(word, text)

    def test_predict_matches_fit_labels(self):
        feats = self.res["report"]["features"]
        elig = self.profiles[self.profiles.active_days >= self.cfg["segmentation"]["min_active_days"]]
        pred = assign_segments(self.res["preprocessor"], self.res["kmeans"], elig, feats)
        np.testing.assert_array_equal(pred, self.res["kmeans"].labels_)
        with self.assertRaises(ValueError):
            assign_segments(self.res["preprocessor"], self.res["kmeans"], elig.drop(columns=[feats[0]]), feats)

    def test_too_few_avatars(self):
        with self.assertRaises(EmptyDatasetError):
            fit_segmentation(self.profiles.head(10), self.cfg)

    def test_negative_feature_rejected(self):
        bad = self.profiles.copy()
        bad.loc[bad.index[0], "est_hours_online"] = -1.0
        with self.assertRaises(ValueError):
            fit_segmentation(bad, self.cfg)

    def test_preprocessor_scales(self):
        feats = self.cfg["segmentation"]["features"]
        elig = self.profiles[self.profiles.active_days >= 3]
        X = make_preprocessor(feats, self.cfg["segmentation"]["log_features"]).fit_transform(elig[feats])
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-8)


class ChurnModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.cfg["models"]["random_forest"]["n_estimators"] = 60   # keep tests quick
        cls.ev = _events(n_avatars=300, days=330, seed=7)
        cls.ds = build_churn_dataset(cls.ev, cls.cfg)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = get_paths(cls.cfg, root=cls.tmp.name)
        cls.out = train_and_evaluate(cls.ds, cls.cfg, cls.paths, data_origin="synthetic_test_fixture")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_models_evaluated_and_metrics_valid(self):
        r = self.out["results"]
        self.assertEqual(set(r["test"]), {"dummy_prior", "recency_only", "logistic_regression", "random_forest"})
        for m in r["test"].values():
            for k in ("precision", "recall", "f1", "accuracy"):
                self.assertTrue(0 <= m[k] <= 1)
            self.assertTrue(0 <= m["roc_auc"] <= 1)
        self.assertAlmostEqual(r["test"]["dummy_prior"]["roc_auc"], 0.5, places=6)
        self.assertIn(r["selected_model"], ("logistic_regression", "random_forest"))

    def test_confusion_matrix_sums_to_test_size(self):
        cm = self.out["results"]["test_selected_tuned_threshold"]["confusion"]
        self.assertEqual(sum(cm.values()), len(self.ds.test))
        self.assertEqual(int(self.out["confusion_matrix"].to_numpy().sum()), len(self.ds.test))

    def test_artifacts_written_and_loadable(self):
        for p in (self.paths.churn_model, self.paths.metrics, self.paths.model_comparison, self.paths.confusion_matrix, self.paths.classification_report):
            self.assertTrue(p.is_file(), p)
        bundle = load_bundle(self.paths)
        self.assertEqual(bundle["meta"]["data_origin"], "synthetic_test_fixture")
        X = self.ds.test[bundle["meta"]["numeric_features"] + bundle["meta"]["categorical_features"]]
        res, warns = predict_churn(bundle, X)
        np.testing.assert_allclose(res["churn_probability"].to_numpy(), self.out["test_proba"], atol=1e-9)

    def test_threshold_selected_without_test_data(self):
        r = self.out["results"]
        self.assertTrue(0 < r["threshold_tuned_on_validation"] <= 1)

    def test_single_class_training_rejected(self):
        ds = self.ds
        broken = type(ds)(ds.spec, ds.plan, ds.snapshots, ds.fit.assign(churned=0), ds.valid, ds.refit, ds.test)
        with self.assertRaises(EmptyDatasetError):
            train_and_evaluate(broken, self.cfg)

    def test_bootstrap_ci_brackets_point_estimate_and_is_by_avatar(self):
        from sklearn.metrics import roc_auc_score
        from src.models.churn import bootstrap_ci
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 600)
        good = y * 0.6 + rng.rand(600) * 0.8          # informative
        noise = rng.rand(600)                          # uninformative
        groups = np.repeat(np.arange(200), 3)          # 3 rows per avatar
        ci = bootstrap_ci(y, {"good": good, "noise": noise}, groups, reference="noise", n_boot=200, seed=1)
        lo, hi = ci["models"]["good"]["roc_auc"]
        self.assertTrue(lo <= roc_auc_score(y, good) <= hi)
        self.assertGreater(ci["roc_auc_difference_vs_reference"]["good"][0], 0)   # clearly better than noise
        self.assertEqual(ci["n_avatars"], 200)

    def test_test_periods_do_not_overlap_training_labels(self):
        d = self.ds
        gap = d.test["cutoff"].min() - pd.Timedelta(days=d.spec.observation_days) - (d.refit["cutoff"].max() + pd.Timedelta(days=d.spec.prediction_days))
        self.assertGreaterEqual(gap, pd.Timedelta(0))
        self.assertEqual(d.test["cutoff"].nunique(), d.spec.n_test_cutoffs)

    def test_metric_helpers(self):
        m = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)
        self.assertEqual(m["confusion"], {"tn": 2, "fp": 0, "fn": 1, "tp": 1})
        self.assertAlmostEqual(m["roc_auc"], 0.75)
        self.assertEqual(best_f1_threshold([0, 0, 0], [0.1, 0.2, 0.3]), 0.5)
        self.assertIsNone(binary_metrics([1, 1], [0.2, 0.9])["roc_auc"])


class PredictionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = {
            "numeric_features": ["active_days", "has_guild"],
            "categorical_features": ["charclass"],
            "feature_rules": {"active_days": [0, 28], "has_guild": [0, 1]},
            "categories_seen": {"charclass": ["Mage", "Rogue"]},
        }

    def row(self, **kw):
        base = {"active_days": 5, "has_guild": 1, "charclass": "Mage"}
        base.update(kw)
        return pd.DataFrame([base])

    def test_valid(self):
        clean, warns = validate_features(self.row(), self.meta)
        self.assertEqual(warns, [])
        self.assertEqual(len(clean), 1)

    def test_out_of_range(self):
        with self.assertRaises(PredictionInputError):
            validate_features(self.row(active_days=99), self.meta)
        with self.assertRaises(PredictionInputError):
            validate_features(self.row(has_guild=-1), self.meta)

    def test_missing_nan_and_text(self):
        with self.assertRaises(PredictionInputError):
            validate_features(self.row().drop(columns=["active_days"]), self.meta)
        with self.assertRaises(PredictionInputError):
            validate_features(self.row(active_days=np.nan), self.meta)
        with self.assertRaises(PredictionInputError):
            validate_features(self.row(active_days="abc"), self.meta)
        with self.assertRaises(PredictionInputError):
            validate_features(self.row(charclass=""), self.meta)

    def test_empty_and_unseen_category(self):
        with self.assertRaises(PredictionInputError):
            validate_features(pd.DataFrame(), self.meta)
        _, warns = validate_features(self.row(charclass="Bard"), self.meta)
        self.assertEqual(len(warns), 1)

    def test_missing_model_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ArtifactMissingError):
                load_bundle(get_paths(load_config(), root=d))


if __name__ == "__main__":
    unittest.main()
