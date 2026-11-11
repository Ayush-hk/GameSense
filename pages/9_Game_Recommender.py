import streamlit as st

from src.dashboard import content, ui
from src.dashboard.games_common import csv, recommender, setup_games
from src.dashboard.profile import load_profile
from src.games import models as M

cfg, out, proc, summary = setup_games("Game recommender", "Pick games you enjoy and see what players with similar tastes also played.")
prof = load_profile()
rec = recommender(proc)
fr = csv(out / "frustration_by_game.csv")
rk = cfg["games"]["recommender"]["k"]

ev = rec["evaluation"]["metrics"]
hit = ev[ev["k"] == rk].set_index("model")["hit_rate"]
main, side = ui.layout()
with side:
    _h = ev[ev["k"] == rk]
    ui.bar_list(f"Hit rate at top {rk}", [(m.replace("_", " ").title(), float(v)) for m, v in zip(_h["model"], _h["hit_rate"])], max_value=1.0, fmt="{:.0%}")
    ui.text_panel("Key insight", ["Suggestions come from players with similar play histories. The hit rate shows how often a game a player really played appeared in the top list, next to simple baselines."], key=True)
with main:
    ui.kpis([
        {"label": "Participants used", "value": f"{rec['evaluation']['n_users']:,}", "sub": "with enough play history", "icon": "👥"},
        {"label": "Games covered", "value": f"{rec['evaluation']['n_games']}", "sub": "played by 10+ people", "icon": "🎮", "tone": "green"},
        {"label": f"Hit rate (top {rk})", "value": f"{hit['item_item']:.0%}", "sub": f"most popular: {hit['popularity']:.0%}  |  random: {hit['random']:.0%}", "icon": "🎯", "tone": "pink"},
    ])

    names = dict(zip(rec["items"], rec["names"]))
    liked = st.multiselect("Games you like", list(rec["items"]), format_func=lambda g: names[g], max_selections=15)
    c = st.columns(2)
    n_rec = c[0].slider("How many to suggest", 3, 20, 10)
    w = c[1].slider("Avoid frustrating games", 0.0, 1.0, 0.0, 0.1,
                    help="0 = plain model. Higher values lower the rank of games that players rate as more frustrating than average.")
    if not liked:
        st.info("Pick at least one game to see suggestions.")
    else:
        out_df = M.recommend(rec, liked, n_rec, frustration=fr, frustration_weight=w)
        show = out_df.rename(columns={"game": "Game", "players": "Players who liked it", "score": "Score"})
        cols = ["Game", "Score", "Players who liked it"] + (["frustration_vs_avg"] if "frustration_vs_avg" in show else [])
        st.dataframe(show[cols].round(3), hide_index=True)
    ui.about_view(prof, content.NOTES["recommender"])
