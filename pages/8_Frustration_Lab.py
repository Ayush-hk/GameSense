import plotly.express as px
import streamlit as st

from src.dashboard import content, ui
from src.dashboard.games_common import csv, pkl, setup_games
from src.dashboard.profile import load_profile
from src.games import analysis as A
from src.games import insights as I

cfg, out, proc, summary = setup_games("Frustration Lab", "Understand which games and genres cause the most frustration, and how different player groups experience it.")
fc = cfg["games"]["frustration"]
k = cfg["games"]["min_group_participants"]
prof = load_profile()
diary = pkl(proc / "diary.pkl")
fr = summary["frustration"]

c = st.columns(4)
plats = sorted(diary["platform"].unique())
platform = c[0].multiselect("Platform", plats, default=plats)
bands = I.order_bands(diary["age_band"].dropna().unique())
start_bands = [b for b in prof["default_age_bands"] if b in bands] or bands
chosen = c[1].multiselect("Age band", bands, default=start_bands)
min_p = c[2].slider("Min participants per game", fc["min_participants_per_game"], 40, fc["min_participants_per_game"])
view = c[3].radio("Measure", ["Shrunken mean", "Versus the player's own average"])
d = diary[diary["platform"].isin(platform) & diary["age_band"].isin(chosen)]
if d.empty:
    st.warning("No diary entries for this selection.")
    st.stop()

label = d.groupby("game_key")["game_label"].agg(lambda s: s.value_counts().index[0])
rank = A.shrunken_frustration(d, min_participants=min_p, min_entries=fc["min_entries_per_game"])
rank["game"] = rank["game_key"].map(label)
by_genre = A.frustration_by_genre(d, pkl(proc / "games.pkl"), k, fc["min_entries_per_game"])
by_platform = A.frustration_by_group(d, "platform", k)
by_age = A.frustration_by_group(d, "age_band", k)
needs = I.need_means(d)
ui.kpis(I.frustration_kpis(d, by_genre, rank))

main, side = st.columns([3, 1.3])
with side:
    ui.bar_list("Top 5 most frustrating titles (avg. score)", [(r.game, r.shrunken_mean) for r in rank.head(5).itertuples()], max_value=7.0)
    ui.text_panel("Quick insights", I.frustration_quick_insights(by_genre, by_platform, by_age, needs), key=True)
    ui.bar_list("Frustration items (avg. 1-7)", needs, max_value=7.0)

with main:
    tab1, tab2, tab3 = st.tabs(["Most frustrating games", "Genre, platform, age", "What kind of frustration"])
    with tab1:
        if view == "Shrunken mean":
            g = rank
            col, lo, hi, title = "shrunken_mean", "ci_low", "ci_high", "Average frustration (1-7)"
        else:
            g = A.within_person_delta(d, min_participants=min_p, min_entries=fc["min_entries_per_game"])
            g["ci_low"], g["ci_high"] = g["delta"] - 1.96 * g["se"], g["delta"] + 1.96 * g["se"]
            g["game"] = g["game_key"].map(label)
            col, lo, hi, title = "delta", "ci_low", "ci_high", "Frustration above the player's own average"
        if g.empty:
            st.warning("No game meets the minimum number of participants in this selection.")
        else:
            n_show = st.slider("Games to show", 5, 40, 20)
            top = g.head(n_show).sort_values(col)
            fig = px.scatter(top, x=col, y="game", error_x=top[hi] - top[col], error_x_minus=top[col] - top[lo],
                             color_discrete_sequence=["#ec4899"], hover_data=["participants", "entries"], labels={col: title, "game": ""})
            ui.chart(fig)
            st.dataframe(g[["game", "participants", "entries", col, lo, hi]].round(2), hide_index=True)
    with tab2:
        a, b = st.columns(2)
        if by_genre.empty:
            a.info("Not enough participants per genre in this selection.")
        else:
            a.plotly_chart(px.bar(by_genre.sort_values("shrunken_mean"), x="shrunken_mean", y="genre", orientation="h",
                                  color_discrete_sequence=["#7c5cff"], hover_data=["participants"]).update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"))
        if by_age.empty:
            b.info("Not enough participants per age band in this selection.")
        else:
            b.plotly_chart(px.bar(by_age, x="age_band", y="mean", error_y=1.96 * by_age["se"], color_discrete_sequence=["#ec4899"],
                                  labels={"mean": "mean frustration"}).update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"))
        if by_platform.empty:
            st.info("Not enough participants per platform in this selection.")
        else:
            ui.chart(px.bar(by_platform, x="platform", y="mean", error_y=1.96 * by_platform["se"], color_discrete_sequence=["#3b82f6"]))
    with tab3:
        needs_tbl = csv(out / "frustration_needs.csv").head(15)
        long = needs_tbl.melt(id_vars=["game_key", "game"], value_vars=["autonomy_frustration", "competence_frustration", "relatedness_frustration"],
                              var_name="item", value_name="score")
        long["item"] = long["item"].map(I.NEED_LABELS)
        ui.chart(px.bar(long, x="score", y="game", color="item", barmode="group", orientation="h"))
        link = summary["steam_reception_link"]
        if link["n_games"] >= 15:
            st.metric("Steam review share vs diary frustration (rank correlation)", f"{link['rho']:.2f}",
                      help=f"{link['n_games']} games with both measures, p={link['p']:.4f}")
    ui.about_view(prof, content.NOTES["frustration"])
