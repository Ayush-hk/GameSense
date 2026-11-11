import streamlit as st

from src.dashboard import ui
from src.dashboard.common import setup
from src.dashboard.profile import ROLES, ProfileError, initials, load_profile, save_profile

cfg, paths = setup("Account & Profile", "Your details for this dashboard. They are saved only on this computer.", "Account")
_flash = st.session_state.pop("gs_flash", None)
if _flash:
    st.success(_flash)
profile = load_profile()

left, right = st.columns([1, 2])
with left:
    org = f" - {ui.esc(profile['organisation'])}" if profile["organisation"] else ""
    st.markdown(
        f'<div class="gs-panel" style="text-align:center"><div class="gs-av" style="width:84px;height:84px;font-size:30px;margin:6px auto 12px">'
        f'{ui.esc(initials(profile["display_name"]))}</div><div class="gs-kv" style="font-size:20px">{ui.esc(profile["display_name"])}</div>'
        f'<div class="gs-kl">{ui.esc(profile["role"])}{org}</div>'
        f'<div class="gs-ks" style="margin-top:8px">{ui.esc(profile["email"]) if profile["email"] else "No e-mail saved"}</div></div>',
        unsafe_allow_html=True,
    )
    ui.note("There is no sign-in: this is a local profile used to personalise the dashboard.")

with right:
    with st.form("profile_form"):
        name = st.text_input("Display name", value=profile["display_name"], max_chars=60)
        role = st.selectbox("Role", ROLES, index=ROLES.index(profile["role"]))
        organisation = st.text_input("Organisation (optional)", value=profile["organisation"], max_chars=80)
        email = st.text_input("E-mail (optional)", value=profile["email"], max_chars=120)
        saved = st.form_submit_button("Save profile")
    if saved:
        try:
            save_profile({**profile, "display_name": name, "role": role, "organisation": organisation, "email": email})
            st.session_state["gs_flash"] = "Profile saved."
            st.rerun()
        except ProfileError as exc:
            st.error(str(exc))
