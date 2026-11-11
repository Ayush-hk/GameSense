import streamlit as st

from src.dashboard import content
from src.dashboard.common import setup

cfg, paths = setup("Terms & Conditions", "Please read these terms before using the dashboard.", "Legal")
app = cfg.get("app", {}) or {}
st.caption(f"Version {app.get('terms_version', '1.0')}  -  Last updated {app.get('terms_date', '')}".strip())
for heading, body in content.terms(cfg):
    st.markdown(f"### {heading}")
    st.markdown(body)
