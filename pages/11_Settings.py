import json

import streamlit as st

from src.dashboard.common import setup
from src.dashboard.profile import ACCENTS, PLATFORMS, ProfileError, load_profile, profile_path, reset_profile, save_profile
from src.games.openplay import age_band_labels

cfg, paths = setup("Settings", "Choose how the dashboard looks and which filters it starts with.", "Account")
_flash = st.session_state.pop("gs_flash", None)
if _flash:
    st.success(_flash)
profile = load_profile()
bands = age_band_labels(cfg["games"]["age_bands"])

with st.form("settings_form"):
    tab1, tab2 = st.tabs(["Appearance", "Defaults"])
    with tab1:
        accent = st.selectbox("Accent colour", list(ACCENTS), index=list(ACCENTS).index(profile["accent"]))
        show_notes = st.checkbox("Show 'About this view' notes on chart pages", value=profile["show_notes"])
    with tab2:
        platform = st.selectbox("Default platform filter", PLATFORMS, index=PLATFORMS.index(profile["default_platform"]))
        age_default = st.multiselect("Default age bands", bands, default=[b for b in profile["default_age_bands"] if b in bands],
                                     help="Leave empty to start with all age bands.")
    saved = st.form_submit_button("Save settings")

if saved:
    try:
        save_profile({**profile, "accent": accent, "show_notes": show_notes, "default_platform": platform, "default_age_bands": age_default})
        st.session_state["gs_flash"] = "Settings saved."
        st.rerun()
    except ProfileError as exc:
        st.error(str(exc))

st.markdown("### Data & privacy")
st.markdown("Your profile is stored only on this computer, in this file:")
st.code(str(profile_path()), language=None)
st.download_button("Download my profile (JSON)", json.dumps(profile, indent=2), file_name="gamesense_profile.json", mime="application/json")
st.markdown("**Reset profile**")
sure = st.checkbox("I want to delete my saved profile and return to the defaults")
if st.button("Reset profile", disabled=not sure):
    reset_profile()
    st.session_state["gs_flash"] = "Profile reset."
    st.rerun()
