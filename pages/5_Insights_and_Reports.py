
import plotly.express as px
import streamlit as st

from src.dashboard import ui
from src.dashboard.common import cget, origin_banner, require, setup
from src.reporting import figures as F
from src.reporting.report import build_insights

cfg, paths = setup("Insights & Reports", badge="Churn project")
require(paths.evaluation_report.is_file(), "No report yet. Run `python -m src.pipeline` with the real dataset in `data/raw/`.")
origin_banner(paths)

main, side = ui.layout()
_ins = build_insights(paths)
with side:
    _find = [i for i in _ins if i["title"] != "Caveat"]
    ui.panel("Key findings", [{"title": i["title"], "text": f"{i['text']} (evidence: {i['evidence']})"} for i in _find])
    for i in _ins:
        if i["title"] == "Caveat":
            ui.text_panel("Read with care", [i["text"]], key=True)
with main:
    groups = cget(paths.churn_by_group)
    if groups is not None and len(groups):
        st.subheader("Observed churn rate by group (descriptive, not causal)")
        dim = st.selectbox("Dimension", sorted(groups["dimension"].unique()))
        g = groups[groups["dimension"] == dim]
        st.plotly_chart(px.bar(g, x="group", y="churn_rate", hover_data=["rows"], title=f"Churn rate by {dim}").update_layout(yaxis_tickformat=".0%", **F.LAYOUT))

    st.subheader("Downloads")
    text = paths.evaluation_report.read_text(encoding="utf-8")
    c = st.columns(4)
    c[0].download_button("Evaluation report (.md)", text.encode("utf-8"), "evaluation_report.md", "text/markdown")
    if paths.metrics.is_file():
        c[1].download_button("Metrics (.json)", paths.metrics.read_bytes(), "metrics.json", "application/json")
    if paths.model_comparison.is_file():
        c[2].download_button("Model comparison (.csv)", paths.model_comparison.read_bytes(), "model_comparison.csv", "text/csv")
    if paths.cluster_profiles.is_file():
        c[3].download_button("Segment profiles (.csv)", paths.cluster_profiles.read_bytes(), "segment_profiles.csv", "text/csv")

    with st.expander("Full evaluation report"):
        st.markdown(text)
