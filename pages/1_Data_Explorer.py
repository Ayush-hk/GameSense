import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import ui
from src.dashboard.common import cget, jget, origin_banner, pget, require, setup
from src.reporting import figures as F

cfg, paths = setup("Data Explorer", badge="Churn project")
profile = jget(paths.dataset_profile)
require(profile is not None)
origin_banner(paths)

main, side = ui.layout()
with side:
    cl = profile["cleaning"]
    ui.text_panel("Key insight", [
        f"The file holds {profile['rows']:,} activity records for {profile['avatars']:,} avatars over {profile['span_days']:.0f} days.",
        f"{cl['rows_removed_total']:,} duplicate or invalid rows were removed during cleaning ({cl['rows_after_cleaning']:,} kept).",
    ], key=True)
    ui.panel("Data checks", [
        {"title": "Timestamps parsed", "text": "Share of timestamps read successfully", "pill": f"{cl['timestamp_parse_rate_overall']:.0%}"},
        {"title": "Levels", "text": f"Range in the data: {profile['level_min']} to {profile['level_max']}"},
        {"title": "Polls per avatar", "text": f"Median {profile['polls_per_avatar_median']:.0f}, maximum {profile['polls_per_avatar_max']:,}"},
    ])
with main:
    ui.kpis([
        {"label": "Avatars", "value": f"{profile['avatars']:,}", "sub": "distinct characters", "icon": "🧙"},
        {"label": "Activity records", "value": f"{profile['rows']:,}", "sub": "after cleaning", "icon": "📈", "tone": "blue"},
        {"label": "Span", "value": f"{profile['span_days']:.0f} days", "sub": f"{profile['date_min'][:10]} to {profile['date_max'][:10]}", "icon": "📅", "tone": "green"},
        {"label": "Columns", "value": f"{len(profile['columns'])}", "sub": "in the raw file", "icon": "🗂️", "tone": "pink"},
    ])
    tab1, tab2, tab3, tab4 = st.tabs(["Schema & quality", "Activity over time", "Avatar distributions", "Cleaning log"])

    with tab1:
        st.dataframe(pd.DataFrame(profile["columns"]), hide_index=True)
        miss = profile["cleaning"]["missing_values_raw"]
        st.write("**Missing values in the raw file**")
        st.dataframe(pd.DataFrame({"column": list(miss), "missing": list(miss.values())}), hide_index=True)

    with tab2:
        daily = cget(paths.daily_activity)
        if daily is not None:
            daily["date"] = pd.to_datetime(daily["date"])
            lo, hi = daily["date"].min().date(), daily["date"].max().date()
            a, b = st.slider("Date range", lo, hi, (lo, hi))
            d = daily[(daily["date"].dt.date >= a) & (daily["date"].dt.date <= b)]
            st.plotly_chart(px.line(d, x="date", y="active_avatars", title="Distinct avatars per day").update_layout(**F.LAYOUT))
        heat = F.hourly_heatmap_fig(paths)
        if heat is not None:
            st.plotly_chart(heat)

    with tab3:
        prof = pget(paths.profiles_table)
        if prof is None:
            st.info("Avatar profiles are produced by the segmentation stage.")
        else:
            f1, f2, f3 = st.columns(3)
            has_class, has_race = "charclass" in prof, "race" in prof
            classes = f1.multiselect("Class", sorted(prof["charclass"].unique())) if has_class else []
            races = f2.multiselect("Race", sorted(prof["race"].unique())) if has_race else []
            if not (has_class or has_race):
                f1.caption("Race and class are not in this data file.")
            lvl = f2.slider("Level at end of window", 1, 80, (1, 80)) if not has_race else (1, 80)
            min_days = f3.slider("Minimum active days", 1, int(prof["active_days"].max()), 1)
            d = prof[(prof["active_days"] >= min_days) & prof["level_last"].between(*lvl)]
            if classes:
                d = d[d["charclass"].isin(classes)]
            if races:
                d = d[d["race"].isin(races)]
            st.caption(f"{len(d):,} of {len(prof):,} avatars match the filters.")
            num = [c for c in ("active_days", "est_hours_online", "mean_session_min", "sessions_per_active_day", "level_last",
                               "zones_visited", "weekend_share", "night_share", "activity_regularity") if c in prof.columns]
            col = st.selectbox("Feature", num)
            log_x = st.checkbox("Log scale (x)", value=col in ("est_hours_online", "mean_session_min"))
            if len(d):
                st.plotly_chart(px.histogram(d, x=col, nbins=60, log_x=log_x, title=f"Distribution of {col}").update_layout(**F.LAYOUT))
                st.dataframe(d[num].describe().T)

    with tab4:
        st.json(profile["cleaning"])
        card = paths.dataset_card
        if card.is_file():
            with st.expander("Generated dataset card"):
                st.markdown(card.read_text(encoding="utf-8"))
