import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import ui
from src.dashboard.common import cget, jget, origin_banner, pget, require, setup
from src.reporting import figures as F

cfg, paths = setup("Player Segmentation", badge="Churn project")
rep = jget(paths.segmentation_report)
require(rep is not None, "Segmentation has not been run yet. Run `python -m src.pipeline --stages segment` (needs the real dataset).")
origin_banner(paths)

main, side = ui.layout()
with side:
    prof = cget(paths.cluster_profiles)
    if prof is not None:
        ui.panel("Segments", [{"title": f"Segment {int(r['segment'])}", "text": r["description"], "pill": f"{r['share']:.0%}"} for _, r in prof.iterrows()])
    ui.text_panel("Key insight", [rep["k_selection_reason"]], key=True)
with main:
    ui.kpis([
        {"label": "Segments (k)", "value": f"{rep['chosen_k']}", "sub": "chosen by validation", "icon": "🧩"},
        {"label": "Silhouette", "value": f"{rep['silhouette_final_sample']:.3f}", "sub": "cluster separation", "icon": "📐", "tone": "blue"},
        {"label": "Seed stability (ARI)", "value": "n/a" if rep.get("stability_ari_mean") is None else f"{rep['stability_ari_mean']:.3f}", "sub": "agreement across seeds", "icon": "🔁", "tone": "green"},
        {"label": "Avatars clustered", "value": f"{rep['avatars_clustered']:,}", "sub": f"{rep['avatars_excluded_insufficient_activity']:,} with under {rep['min_active_days']} active days left out", "icon": "👥", "tone": "pink"},
    ])
    if rep["silhouette_final_sample"] < 0.25:
        ui.note("The segments overlap, so read them as a descriptive summary rather than sharply separated player types.")
    fig = F.cluster_selection_fig(paths)
    if fig is not None:
        st.plotly_chart(fig)

    st.subheader("Segment profiles (descriptions derived only from standardised feature differences)")
    if prof is not None:
        st.dataframe(prof.drop(columns=["description"]), hide_index=True)
        for _, r in prof.iterrows():
            st.markdown(f"**Segment {int(r['segment'])}** ({r['share']:.1%} of clustered avatars): {r['description']}")

    z = pd.DataFrame(rep["z_means"]).T
    z.index.name = "segment"
    st.plotly_chart(px.imshow(z.to_numpy(), x=list(z.columns), y=[f"Segment {i}" for i in z.index], color_continuous_scale="RdBu_r",
                              color_continuous_midpoint=0, aspect="auto", title="Mean standardised feature value per segment (SD from population mean)").update_layout(**F.LAYOUT))

    pts = pget(paths.profiles_table)
    if pts is not None:
        pts = pts[pts["segment"] >= 0]
        n = st.slider("Points to plot", 1000, min(30000, max(1000, len(pts))), min(8000, len(pts)), step=500)
        samp = pts.sample(n=min(n, len(pts)), random_state=0).assign(segment=lambda d: d["segment"].astype(str))
        ev = rep["pca_explained_variance_ratio"]
        st.plotly_chart(px.scatter(samp, x="pc1", y="pc2", color="segment", opacity=0.5, hover_data=["active_days", "est_hours_online", "level_last"],
                                   title=f"PCA projection (PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%} of variance) - a 2-D view, not the clustering space").update_layout(**F.LAYOUT))
        if "charclass" in pts:
            mix = pts.groupby(["segment", "charclass"]).size().rename("avatars").reset_index()
            mix["segment"] = mix["segment"].astype(str)
            st.plotly_chart(px.bar(mix, x="segment", y="avatars", color="charclass", barmode="relative", title="Class mix per segment").update_layout(**F.LAYOUT))
        else:
            lv = pts.assign(segment=pts["segment"].astype(str))
            st.plotly_chart(px.box(lv, x="segment", y="level_last", title="Level at end of observation, by segment").update_layout(**F.LAYOUT))
