import streamlit as st

from src.dashboard import content
from src.dashboard.common import setup

cfg, paths = setup("Privacy Policy", "What the dashboard stores and how the datasets are protected.", "Legal")
app = cfg.get("app", {}) or {}
st.caption(f"Version {app.get('terms_version', '1.0')}  -  Last updated {app.get('terms_date', '')}".strip())
for heading, body in content.privacy(cfg):
    st.markdown(f"### {heading}")
    st.markdown(body)
