"""Executes the Game Lab pages with stand-in streamlit/plotly modules.

This proves the page code runs against the real pipeline outputs (no NameError, bad column, empty-frame crash).
It does NOT prove anything about how the pages look or behave in a browser: streamlit and plotly were not installable
in the build environment. Skipped when the pipeline has not been run.
"""
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["6_Games_Played", "7_Age_and_Games", "8_Frustration_Lab", "9_Game_Recommender", "10_Game_Models",
         "11_Settings", "12_Account_and_Profile", "13_Terms_and_Conditions", "14_Privacy_Policy", "15_Help_and_FAQ"]


class StopPage(Exception):
    pass


class FakeSt:
    """Returns sensible widget values so page logic follows its normal path; everything else is a no-op."""
    calls: list = []

    def __init__(self):
        self.stopped = False

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def columns(self, n, **kw):
        return [FakeSt() for _ in range(n if isinstance(n, int) else len(n))]

    def tabs(self, labels): return [FakeSt() for _ in labels]
    def selectbox(self, label, options, *a, **kw): return list(options)[0]
    def radio(self, label, options, *a, **kw): return list(options)[0]
    def checkbox(self, label, value=False, **kw): return value

    def multiselect(self, label, options, default=None, **kw):
        return list(default) if default is not None else list(options)[:2]

    def slider(self, label, lo=0, hi=1, value=None, *a, **kw):
        return lo if value is None else value

    def stop(self): raise StopPage()

    # forms and buttons are never "pressed" in tests, so no page writes to the real profile file
    def button(self, *a, **kw): return False
    def form_submit_button(self, *a, **kw): return False
    def text_input(self, label, value="", *a, **kw): return value
    def form(self, *a, **kw): return FakeSt()
    def expander(self, *a, **kw): return FakeSt()
    session_state = {}

    def cache_data(self, *a, **kw):
        return a[0] if a and callable(a[0]) else (lambda f: f)
    cache_resource = cache_data

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        FakeSt.calls.append(name)
        return MagicMock()


def run_page(name):
    st = FakeSt()
    px = MagicMock()
    mods = {"streamlit": st, "plotly": MagicMock(), "plotly.express": px, "plotly.graph_objects": MagicMock()}
    saved = {k: sys.modules.get(k) for k in list(mods) + [m for m in sys.modules if m.startswith("src.dashboard")]}
    for k in [m for m in sys.modules if m.startswith("src.dashboard")]:
        del sys.modules[k]
    sys.modules.update(mods)
    try:
        try:
            runpy.run_path(str(ROOT / "app.py" if name == "app" else ROOT / "pages" / f"{name}.py"), run_name="__main__")
        except StopPage:
            return "stopped"
        return "ok"
    finally:
        for k in [m for m in sys.modules if m.startswith("src.dashboard")]:
            del sys.modules[k]
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class FuzzSt(FakeSt):
    """Random widget values, so filters that empty a table or pick unusual options are exercised."""
    rng = __import__("random").Random(0)

    def columns(self, n, **kw): return [FuzzSt() for _ in range(n if isinstance(n, int) else len(n))]
    def tabs(self, labels): return [FuzzSt() for _ in labels]
    def selectbox(self, label, options, *a, **kw): return self.rng.choice(list(options))
    def radio(self, label, options, *a, **kw): return self.rng.choice(list(options))
    def checkbox(self, label, value=False, **kw): return self.rng.random() < 0.5
    def text_input(self, label, value="", *a, **kw): return self.rng.choice([value, "", "balder", "xyzzy"])

    def multiselect(self, label, options, default=None, **kw):
        o = list(options)
        k = self.rng.randint(0, len(o))
        if kw.get("max_selections"):
            k = min(k, kw["max_selections"])
        return self.rng.sample(o, k)

    def slider(self, label, lo=0, hi=1, value=None, *a, **kw):
        import datetime as _dt
        if isinstance(lo, _dt.date):
            d = self.rng.randint(0, max((hi - lo).days, 0))
            e = self.rng.randint(d, max((hi - lo).days, 0))
            r = (lo + _dt.timedelta(days=d), lo + _dt.timedelta(days=e))
            return r if isinstance(value, tuple) else r[0]
        if isinstance(lo, float) or isinstance(hi, float):
            v = [round(self.rng.uniform(lo, hi), 1) for _ in range(2)]
        else:
            v = [self.rng.randint(lo, hi) for _ in range(2)]
        return tuple(sorted(v)) if isinstance(value, tuple) else v[0]


def run_fuzzed(name, seed):
    global FakeSt
    import random
    FuzzSt.rng = random.Random(seed)
    saved = FakeSt
    FakeSt = FuzzSt
    try:
        return run_page(name)
    finally:
        FakeSt = saved


@unittest.skipUnless((ROOT / "reports" / "games" / "summary.json").is_file(), "run python -m src.games.pipeline first")
class PageSmokeTests(unittest.TestCase):
    def test_pages_run_to_the_end(self):
        for name in PAGES:
            with self.subTest(page=name):
                self.assertEqual(run_page(name), "ok")

    def test_pages_survive_random_filter_choices(self):
        # regression: page 6 crashed on genre filters (NaN genres), page 8 on filters that left no genre
        for name in PAGES:
            for seed in range(25):
                with self.subTest(page=name, seed=seed):
                    self.assertIn(run_fuzzed(name, seed), ("ok", "stopped"))


@unittest.skipUnless((ROOT / "data" / "processed" / "events.pkl").is_file() or (ROOT / "reports" / "evaluation").is_dir(), "run python -m src.pipeline first")
class ChurnPageSmokeTests(unittest.TestCase):
    """Overview and the original churn pages (1-5) after the redesign."""

    def test_overview_and_churn_pages_run(self):
        from src.config import get_paths, load_config
        paths = get_paths(load_config())
        if not paths.dataset_profile.is_file():
            self.skipTest("churn pipeline results not present")
        for name in ["app", "0_Overview", "1_Data_Explorer", "2_Player_Segmentation", "3_Prediction_Lab", "4_Explainability", "5_Insights_and_Reports"]:
            for seed in range(8):
                with self.subTest(page=name, seed=seed):
                    self.assertIn(run_fuzzed(name, seed) if seed else run_page(name), ("ok", "stopped"))


def _raise_like_old_pandas():
    raise TypeError("StringDtype.__init__() takes from 1 to 2 positional arguments but 3 were given")


class _Unreadable:
    def __reduce__(self):
        return (_raise_like_old_pandas, ())


class DataReadFailureTests(unittest.TestCase):
    """A saved file that the installed pandas cannot read must give a plain explanation, not a traceback."""

    def test_unreadable_pickle_shows_environment_help(self):
        import pickle, tempfile
        from pathlib import Path
        st = FakeSt()
        seen = []
        st.error = lambda msg, *a, **k: seen.append(msg)
        saved = {k: sys.modules.get(k) for k in ["streamlit"] + [m for m in sys.modules if m.startswith("src.dashboard")]}
        for k in [m for m in sys.modules if m.startswith("src.dashboard")]:
            del sys.modules[k]
        sys.modules["streamlit"] = st
        try:
            from src.dashboard import games_common as gc
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "x.pkl"
                f.write_bytes(pickle.dumps(_Unreadable()))
                with self.assertRaises(StopPage):
                    gc.pkl(f)
            self.assertTrue(seen and "pandas" in seen[0] and "requirements.txt" in seen[0])
        finally:
            for k in [m for m in sys.modules if m.startswith("src.dashboard")]:
                del sys.modules[k]
            for k, v in saved.items():
                if v is not None:
                    sys.modules[k] = v
                else:
                    sys.modules.pop(k, None)


if __name__ == "__main__":
    unittest.main()
