"""Profile store, insight text, page content and HTML escaping (no Streamlit needed)."""
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard import content
from src.dashboard.profile import (DEFAULTS, ProfileError, initials, load_profile, reset_profile, save_profile, validate)
from src.games import insights as I

ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "user" / "profile.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_gives_defaults(self):
        self.assertEqual(load_profile(self.path), DEFAULTS)

    def test_save_load_roundtrip_and_reset(self):
        saved = save_profile({"display_name": "  Asha Rao ", "role": "Researcher", "email": "a@b.co", "accent": "Teal",
                              "default_platform": "Steam", "default_age_bands": ["18-22"], "show_notes": False}, self.path)
        self.assertEqual(saved["display_name"], "Asha Rao")
        self.assertEqual(load_profile(self.path), saved)
        reset_profile(self.path)
        self.assertEqual(load_profile(self.path), DEFAULTS)

    def test_invalid_values_are_rejected_and_nothing_is_written(self):
        bad = [{"display_name": ""}, {"display_name": "x" * 61}, {"display_name": "a\x00b"}, {"role": "Wizard"},
               {"email": "not-an-email"}, {"accent": "Green"}, {"default_platform": "Atari"}, {"default_age_bands": "18-22"}]
        for b in bad:
            with self.subTest(b=b):
                with self.assertRaises(ProfileError):
                    save_profile(b, self.path)
        self.assertFalse(self.path.exists())

    def test_damaged_file_falls_back_without_crashing(self):
        self.path.parent.mkdir(parents=True)
        for text in ("{not json", "[1,2,3]", '{"display_name": 5, "role": "Wizard", "accent": "Pink"}'):
            self.path.write_text(text, encoding="utf-8")
            p = load_profile(self.path)
            self.assertIn(p["role"], ("Student", "Researcher", "Analyst", "Educator", "Other"))
        self.assertEqual(load_profile(self.path)["accent"], "Pink")      # valid field kept

    def test_unknown_keys_dropped(self):
        clean, errs = validate({"display_name": "Sam", "is_admin": True})
        self.assertEqual(errs, [])
        self.assertNotIn("is_admin", clean)

    def test_initials(self):
        self.assertEqual(initials("Asha Rao"), "AR")
        self.assertEqual(initials("Prince"), "PR")
        self.assertEqual(initials("  "), "G")
        self.assertEqual(initials("a b c"), "AC")


class InsightTests(unittest.TestCase):
    def matrix(self):
        rows = []
        for band, genre, lift, players in [("18-22", "Platformer", 1.5, 100), ("36+", "Platformer", 0.6, 80),
                                            ("18-22", "Family", 0.2, 50), ("36+", "Family", 5.0, 5),     # 5 players: too few
                                            ("18-22", "RPG", 0.8, 90), ("36+", "RPG", 1.3, 90)]:
            rows.append({"age_band": band, "genre": genre, "lift": lift, "players": players})
        return pd.DataFrame(rows)

    def test_small_cells_cannot_headline(self):
        out = I.age_genre_insights(self.matrix(), n=5, min_players=30)
        self.assertNotIn("Family", [r["title"] for r in out])
        self.assertEqual(out[0]["title"], "Platformer")
        self.assertEqual(out[0]["pill"], "+50%")

    def test_key_insight_sentence(self):
        s = I.age_key_insight(self.matrix())
        self.assertIn("18-22", s)
        self.assertIn("Platformer", s)
        self.assertIn("RPG", s)
        self.assertNotIn("Family", s)

    def test_band_order_is_numeric_not_alphabetical(self):
        self.assertEqual(I.order_bands(["36+", "18-22", "27-30", "23-26"]), ["18-22", "23-26", "27-30", "36+"])

    def test_compact(self):
        self.assertEqual(I.compact(950), "950")
        self.assertEqual(I.compact(1500), "1.5k")
        self.assertEqual(I.compact(22848), "23k")
        self.assertEqual(I.compact(2_500_000), "2.5M")

    def test_frustration_insights_use_given_tables(self):
        g = pd.DataFrame({"genre": ["MOBA", "Shooter", "Puzzle"], "shrunken_mean": [3.0, 2.0, 1.0]})
        pl = pd.DataFrame({"platform": ["Steam", "Xbox"], "mean": [2.5, 2.0]})
        ag = pd.DataFrame({"age_band": ["18-22", "36+"], "mean": [2.6, 2.0]})
        lines = I.frustration_quick_insights(g, pl, ag, [("Toxic interactions with others", 2.4)])
        text = " ".join(lines)
        self.assertIn("MOBA and Shooter", text)
        self.assertIn("2.50", text)
        self.assertIn("Steam players report the highest", text)
        self.assertIn("18-22", text)

    def test_empty_inputs_give_no_lines_and_no_crash(self):
        e = pd.DataFrame(columns=["genre", "shrunken_mean"])
        self.assertEqual(I.frustration_quick_insights(e, pd.DataFrame(), pd.DataFrame(), []), [])


class ContentTests(unittest.TestCase):
    def test_author_and_contact_only_when_configured(self):
        empty = {"app": {"author": "", "contact": ""}}
        full = {"app": {"author": "Asha Rao, Example University", "contact": "asha@example.org"}}
        t_empty = " ".join(b for _, b in content.terms(empty))
        t_full = " ".join(b for _, b in content.terms(full))
        self.assertNotIn("asha@example.org", t_empty)
        self.assertIn("Asha Rao, Example University", t_full)
        self.assertIn("asha@example.org", t_full)
        self.assertIn("project's author(s)", t_empty)

    def test_terms_keep_the_reidentification_rule(self):
        t = " ".join(b for _, b in content.terms({}))
        self.assertIn("re-identification", t.lower())
        self.assertIn("identify", t)

    def test_privacy_claim_matches_the_actual_config(self):
        text = " ".join(b for _, b in content.privacy({}))
        self.assertIn("usage statistics", text)
        cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        self.assertIn("gatherUsageStats = false", cfg)

    def test_faq_is_non_empty_and_unique(self):
        qs = [q for _, items in content.FAQ for q, _ in items]
        self.assertGreater(len(qs), 8)
        self.assertEqual(len(qs), len(set(qs)))


class EscapingTests(unittest.TestCase):
    """Data-derived text (for example a game name) must never be able to inject markup."""

    def setUp(self):
        self.out = []
        fake = types.ModuleType("streamlit")
        fake.markdown = lambda html, **kw: self.out.append(html)
        fake.sidebar = types.SimpleNamespace(markdown=lambda html, **kw: self.out.append(html))
        self.saved = {k: sys.modules.get(k) for k in ("streamlit", "src.dashboard.ui")}
        sys.modules["streamlit"] = fake
        sys.modules.pop("src.dashboard.ui", None)
        self.ui = importlib.import_module("src.dashboard.ui")

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_kpis_panel_and_bars_escape_html(self):
        evil = '<img src=x onerror="alert(1)">'
        self.ui.kpis([{"label": evil, "value": evil, "sub": evil, "icon": "x"}])
        self.ui.panel(evil, [{"title": evil, "text": evil, "pill": evil}])
        self.ui.bar_list(evil, [(evil, 2.0)])
        self.ui.text_panel(evil, [evil])
        self.ui.hero(evil, evil, evil, [(evil, evil)])
        self.ui.sidebar_card({"display_name": evil, "role": evil})
        joined = "".join(self.out)
        self.assertNotIn("<img", joined)
        self.assertIn("&lt;img", joined)

    def test_bar_widths_are_bounded(self):
        self.ui.bar_list("t", [("a", 1.0), ("b", 0.0), ("c", 99.0)], max_value=7.0)
        html = "".join(self.out)
        self.assertNotIn("width:-", html)
        self.assertNotIn("nan", html.lower())


class NavigationTests(unittest.TestCase):
    def test_every_page_file_is_in_the_navigation_and_exists(self):
        import re
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        listed = set(re.findall(r'page\("([^"]+\.py)"', (root / "app.py").read_text(encoding="utf-8")))
        on_disk = {p.name for p in (root / "pages").glob("*.py")}
        self.assertEqual(listed, on_disk)
        self.assertTrue((root / "assets" / "logo.svg").is_file())


if __name__ == "__main__":
    unittest.main()
