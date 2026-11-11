import plotly.express as px
import streamlit as st

from src.dashboard import content, ui
from src.dashboard.games_common import pkl, setup_games
from src.dashboard.profile import PLATFORMS, load_profile
from src.games import analysis as A
from src.games import insights as I

cfg, out, proc, summary = setup_games("What people play", "See which games and genres people spend their time on, across platforms.")
k = cfg["games"]["min_group_participants"]
prof = load_profile()
play = pkl(proc / "playtime.pkl")
people = pkl(proc / "participants.pkl")

bands = I.order_bands(people["age_band"].dropna().unique())
c = st.columns(4)
platform = c[0].selectbox("Platform", PLATFORMS, index=PLATFORMS.index(prof["default_platform"]))
start_bands = [b for b in prof["default_age_bands"] if b in bands] or bands
chosen = c[1].multiselect("Age band", bands, default=start_bands)
genre = c[2].selectbox("Genre", ["All"] + sorted(A.genre_popularity(play)["genre"]))
top_n = c[3].slider("Games to show", 5, 30, 15)

d = play.merge(people[["pid", "age_band"]], on="pid")
d = d[d["age_band"].isin(chosen)]
if platform != "All":
    d = d[d["platform"] == platform]
if genre != "All":
    d = d[d["genres_clean"].map(lambda s: genre in A.harmonise_genres(s))]
if d.empty:
    st.warning("No play data for this combination of filters.")
    st.stop()

most = A.most_played(d, None, min_players=k)
main, side = ui.layout()
with side:
    _gp = A.genre_popularity(d).head(5)
    ui.bar_list("Top genres (share of hours)", list(zip(_gp["genre"], _gp["share_of_hours"])), fmt="{:.0%}")
    if not most.empty:
        _top = most.sort_values("hours", ascending=False).head(5)
        ui.bar_list("Most played (total hours)", list(zip(_top["game"], _top["hours"])), fmt="{:,.0f} h")
with main:
    ui.kpis(I.played_kpis(d, most))
    tab1, tab2 = st.tabs(["Most played games", "Genres"])
    with tab1:
        if platform == "Xbox":
            st.info("Xbox titles are not named in the data, only their genre, so there is no game ranking for Xbox. See the Genres tab.")
        elif most.empty:
            st.warning(f"No game has at least {k} players in this selection.")
        else:
            metric = st.radio("Rank by", ["Total hours", "Number of players"], horizontal=True)
            col = "hours" if metric == "Total hours" else "players"
            shown = most.sort_values(col, ascending=False).head(top_n)
            fig = px.bar(shown.sort_values(col), x=col, y="game", orientation="h", color_discrete_sequence=["#7c5cff"],
                         hover_data=["players", "median_hours_per_player", "genres"],
                         labels={"hours": "Total hours played", "players": "Participants who played it", "game": ""})
            ui.chart(fig)
            st.dataframe(shown[["game", "hours", "players", "median_hours_per_player", "share_of_named_hours"]].round(2), hide_index=True)
    with tab2:
        gp = A.genre_popularity(d).head(20)
        fig = px.bar(gp.sort_values("share_of_hours"), x="share_of_hours", y="genre", orientation="h", color_discrete_sequence=["#7c5cff"],
                     hover_data=["players"], labels={"share_of_hours": "Share of hours", "genre": ""})
        ui.chart(fig)
    ui.about_view(prof, content.NOTES["played"])
