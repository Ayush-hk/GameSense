"""GameSense - Overview page (opened from the app.py navigation)."""
import streamlit as st

from src.dashboard import ui
from src.dashboard.common import jget, origin_banner, require, setup
from src.reporting import figures as F

cfg, paths = setup("Overview", "Explainable analytics, segmentation and churn prediction for video-game player behaviour.", "GameSense")

profile = jget(paths.dataset_profile)
require(profile is not None)
info = origin_banner(paths)

ui.kpis([
    {"label": "Avatars analysed", "value": f"{profile['avatars']:,}", "sub": "World of Warcraft telemetry", "icon": "🧙"},
    {"label": "Activity records", "value": f"{profile['rows']:,}", "sub": "10-minute polls", "icon": "📈", "tone": "blue"},
    {"label": "Span", "value": f"{profile['span_days']:.0f} days", "sub": f"{profile['date_min'][:10]} to {profile['date_max'][:10]}", "icon": "📅", "tone": "green"},
    {"label": "Source", "value": "WoWAH", "sub": "Lee, Chen, Cheng and Lei, 2011", "icon": "📚", "tone": "pink"},
])

main, side = st.columns([3, 1.2])
with main:
    fig = F.daily_activity_fig(paths)
    if fig is not None:
        st.plotly_chart(fig)
with side:
    status = jget(paths.pipeline_status) or {}
    rows = [{"title": s.replace("_", " ").title(), "text": v.get("finished_at") or v.get("message", ""),
             "pill": v.get("status", "not run")} for s, v in status.items()]
    if rows:
        ui.panel("Analysis status", rows)
    ui.text_panel("Where to start", [
        "Pages 1-5: the churn project (segments, prediction, explanations).",
        "Pages 6-10: the Game Lab (what people play, age, frustration, recommender).",
        "Settings, profile, terms, privacy and help are at the end of the sidebar.",
    ])
if info:
    st.caption(f"Data file `{info['raw_file']['file_name']}` - SHA-256 `{info['raw_file']['sha256'][:16]}...` - run at {info['created_at']}")
