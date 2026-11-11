import plotly.express as px
import streamlit as st

from src.dashboard import content, ui
from src.dashboard.games_common import csv, pkl, setup_games
from src.dashboard.profile import load_profile
from src.games import analysis as A
from src.games import insights as I

cfg, out, proc, summary = setup_games("Age and Games", "Explore how different age groups engage with game genres and which titles stand out.")
k = cfg["games"]["min_group_participants"]
prof = load_profile()
play = pkl(proc / "playtime.pkl")
people = pkl(proc / "participants.pkl")

mat = A.age_genre_lift(A.age_genre_matrix(play, people, k))
main, side = st.columns([3, 1.25])

with side:
    ui.text_panel("Key insight", [I.age_key_insight(mat)], key=True)
    ui.panel("Genre insights", I.age_genre_insights(mat))

with main:
    ui.kpis(I.age_kpis(people, play, A.genre_popularity(play)))
    tab1, tab2, tab3 = st.tabs(["Genre mix by age", "Top games in an age band", "Does age change genre taste?"])
    with tab1:
        how = st.radio("Show", ["Share of the band's play time", "Lift versus all ages"], horizontal=True)
        n_g = st.slider("Genres", 5, 20, 10)
        keep = mat.groupby("genre")["hours"].sum().sort_values(ascending=False).head(n_g).index
        val = "share" if how.startswith("Share") else "lift"
        bands = I.order_bands(mat["age_band"])
        pv = mat[mat["genre"].isin(keep)].pivot(index="genre", columns="age_band", values=val).reindex(index=keep, columns=bands)
        fig = px.imshow(pv, text_auto=".2f", aspect="auto", color_continuous_scale="Blues" if val == "share" else "RdBu_r",
                        color_continuous_midpoint=None if val == "share" else 1.0, labels={"color": val})
        ui.chart(fig)
    with tab2:
        band = st.selectbox("Age band", I.order_bands(people["age_band"].dropna().unique()))
        top = A.top_games_by_age(play, people, band, 15, k)
        st.write(f"{(people['age_band'] == band).sum():,} participants in this band.")
        if top.empty:
            st.warning("Not enough players per game in this band.")
        else:
            fig = px.bar(top.sort_values("hours"), x="hours", y="game", orientation="h", color="lift", color_continuous_scale="RdBu_r",
                         color_continuous_midpoint=1.0, hover_data=["players"])
            ui.chart(fig)
    with tab3:
        tr = csv(out / "age_genre_trend.csv")
        show = st.checkbox("Only differences that hold after correction for multiple comparisons", value=True)
        t = tr[tr["significant"]] if show else tr
        fig = px.bar(t.sort_values("rho"), x="rho", y="genre", orientation="h", color="significant",
                     labels={"rho": "Rank correlation: age vs share of a person's play time"})
        ui.chart(fig)
    ui.about_view(prof, content.NOTES["age"])
