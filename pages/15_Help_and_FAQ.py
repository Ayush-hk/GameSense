import streamlit as st

from src.dashboard import content
from src.dashboard.common import setup

cfg, paths = setup("Help & FAQ", "Answers about the data, the measures and how to use each page.", "Help")
query = st.text_input("Search help topics", placeholder="for example: frustration, lift, age").strip().lower()

shown = 0
for category, items in content.FAQ:
    hits = [(q, a) for q, a in items if not query or query in q.lower() or query in a.lower()]
    if not hits:
        continue
    st.markdown(f"### {category}")
    for q, a in hits:
        with st.expander(q):
            st.markdown(a)
        shown += 1
if shown == 0:
    st.info("No help topic matches your search. Try a shorter word.")
contact = (cfg.get("app", {}) or {}).get("contact", "").strip()
if contact:
    st.markdown("### Still need help?")
    st.markdown(f"Contact: {contact}")
