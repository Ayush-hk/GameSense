"""GameSense - launch with:  streamlit run app.py

This file only builds the sidebar navigation; every page is a script in pages/.
"""
import importlib.util
import sys
from pathlib import Path

import streamlit as st

_NEEDED = {"yaml": "PyYAML", "plotly": "plotly", "sklearn": "scikit-learn", "scipy": "scipy", "joblib": "joblib", "numpy": "numpy", "pandas": "pandas"}


def _is_missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except (ValueError, ImportError):   # module object present but without import metadata: it is loaded, so not missing
        return False


_missing = [pkg for mod, pkg in _NEEDED.items() if _is_missing(mod)]
if _missing:
    st.error(f"This Python is missing: {', '.join(_missing)}.  Install everything GameSense needs with this command, then reload:\n\n"
             f'`"{sys.executable}" -m pip install -r requirements.txt`')
    st.stop()

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / "assets" / "logo.svg"


def page(name: str, title: str, icon: str, default: bool = False):
    return st.Page(str(ROOT / "pages" / name), title=title, icon=icon, default=default)


if LOGO.is_file():
    st.logo(str(LOGO), size="large")

NAV = {
    "": [page("0_Overview.py", "Overview", ":material/home:", default=True)],
    "Churn project": [
        page("1_Data_Explorer.py", "1  Data Explorer", ":material/search:"),
        page("2_Player_Segmentation.py", "2  Player Segmentation", ":material/hub:"),
        page("3_Prediction_Lab.py", "3  Prediction Lab", ":material/science:"),
        page("4_Explainability.py", "4  Explainability", ":material/lightbulb:"),
        page("5_Insights_and_Reports.py", "5  Insights & Reports", ":material/description:"),
    ],
    "Game Lab": [
        page("6_Games_Played.py", "6  Games Played", ":material/sports_esports:"),
        page("7_Age_and_Games.py", "7  Age and Games", ":material/bar_chart:"),
        page("8_Frustration_Lab.py", "8  Frustration Lab", ":material/mood_bad:"),
        page("9_Game_Recommender.py", "9  Game Recommender", ":material/recommend:"),
        page("10_Game_Models.py", "10  Game Models", ":material/model_training:"),
    ],
    "Account": [
        page("11_Settings.py", "Settings", ":material/settings:"),
        page("12_Account_and_Profile.py", "Account & Profile", ":material/person:"),
        page("13_Terms_and_Conditions.py", "Terms & Conditions", ":material/gavel:"),
        page("14_Privacy_Policy.py", "Privacy Policy", ":material/shield:"),
        page("15_Help_and_FAQ.py", "Help & FAQ", ":material/help:"),
    ],
}

st.navigation(NAV).run()
