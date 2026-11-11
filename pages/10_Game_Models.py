import plotly.express as px
import streamlit as st

from src.dashboard import content, ui
from src.dashboard.games_common import csv, setup_games
from src.dashboard.profile import load_profile
from src.games import insights as I

cfg, out, proc, summary = setup_games("Game models", "How well each model performs on data it did not train on, next to simple baselines.")
prof = load_profile()
frus = csv(out / "model_frustration_metrics.csv")
rec = csv(out / "model_recommender_metrics.csv")
rcp = csv(out / "model_reception_metrics.csv")
main, side = ui.layout()
with side:
    _r = rec[rec["k"] == cfg["games"]["recommender"]["k"]] if "k" in rec else rec
    if "hit_rate" in _r and "model" in _r:
        ui.bar_list(f"Recommender hit rate at top {cfg['games']['recommender']['k']}", [(m.replace("_", " ").title(), float(v)) for m, v in zip(_r["model"], _r["hit_rate"])], max_value=1.0, fmt="{:.0%}")
    ui.text_panel("Key insight", ["Every model is scored on data it did not train on and shown next to a simple baseline, so you can see how much it adds."], key=True)
with main:
    ui.kpis(I.model_kpis(frus, rec, rcp, cfg["games"]["recommender"]["k"]))

    tab1, tab2, tab3 = st.tabs(["Frustration", "Steam reception", "Recommender"])
    with tab1:
        s = summary["models"]["frustration"]
        st.write(f"Predict whether a diary entry is high-frustration: {s['n_entries']:,} entries from {s['n_participants']} participants, "
                 f"{s['prevalence']:.0%} high. Each participant is entirely in training or testing; intervals resample participants.")
        st.dataframe(frus.round(3), hide_index=True)
        ui.note("Most of the predictive signal comes from a player's own earlier ratings (person_only); knowing the game adds a smaller amount.")
        co = csv(out / "model_frustration_coefficients.csv")
        ui.chart(px.bar(co, x=co.columns[1], y="feature", orientation="h", color_discrete_sequence=["#7c5cff"]))
    with tab2:
        s = summary["models"]["reception"]
        st.write(f"Predict whether a Steam game reaches 80% positive reviews from price, release year, genres and tags. "
                 f"Trained on releases up to 2022 ({s['n_train']:,} games), tested on later releases ({s['n_test']:,}). "
                 f"Positive rate: {s['positive_rate_train']:.0%} in training, {s['positive_rate_test']:.0%} in testing.")
        st.dataframe(rcp.round(3), hide_index=True)
        imp = csv(out / "model_reception_importance.csv")
        ui.chart(px.bar(imp.sort_values(imp.columns[1]), x=imp.columns[1], y="feature", orientation="h", color_discrete_sequence=["#7c5cff"]))
    with tab3:
        st.dataframe(rec.round(3), hide_index=True)
        ui.chart(px.bar(rec, x="k", y="hit_rate", color="model", barmode="group"))
    ui.about_view(prof, content.NOTES["models"])
