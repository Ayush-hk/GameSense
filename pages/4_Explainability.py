import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import ui
from src.dashboard.common import cget, get_bundle, jget, origin_banner, pget, require, setup
from src.explainability.explain import CAUSALITY_WARNING, ShapExplainer, shap_available
from src.reporting import figures as F

cfg, paths = setup("Explainability", badge="Churn project")
rep = jget(paths.explain_report)
require(rep is not None, "Explanations have not been generated yet. Run the pipeline on the real dataset first (`python -m src.pipeline`).")
origin_banner(paths)

main, side = ui.layout()
with side:
    _shap, _perm = cget(paths.shap_global), cget(paths.permutation_importance)
    if _shap is not None and len(_shap):
        _t = _shap.head(5)
        ui.bar_list("Top 5 features (mean |SHAP|)", list(zip(_t["feature"], _t["mean_abs_shap"])), fmt="{:.3f}")
    elif _perm is not None and len(_perm):
        _t = _perm.head(5)
        ui.bar_list("Top 5 features (permutation importance)", list(zip(_t["feature"], _t["importance_mean"])), fmt="{:.3f}")
    ui.text_panel("Key insight", [CAUSALITY_WARNING], key=True)
with main:
    ui.kpis([
        {"label": "Method", "value": str(rep["method"]), "sub": "how importance is measured", "icon": "🔍"},
        {"label": "Features ranked", "value": str(0 if cget(paths.shap_global) is None and cget(paths.permutation_importance) is None else len(cget(paths.shap_global) if cget(paths.shap_global) is not None else cget(paths.permutation_importance))), "sub": "in the importance tables", "icon": "📊", "tone": "blue"},
    ])
    st.caption(f"Method: **{rep['method']}**. {rep.get('note', '')}")

    tab1, tab2 = st.tabs(["Global importance", "Explain one prediction"])

    with tab1:
        shap_df = cget(paths.shap_global)
        perm = cget(paths.permutation_importance)
        if shap_df is not None:
            st.plotly_chart(F.importance_fig(shap_df, "mean_abs_shap", f"Mean |SHAP| ({rep.get('units', '')})"))
            sample = cget(paths.shap_sample)
            if sample is not None:
                feat = st.selectbox("Feature dependence", list(shap_df["feature"]))
                vcol = f"value__{feat}"
                scol = f"shap__{feat}"
                if vcol in sample and pd.api.types.is_numeric_dtype(sample[vcol]):
                    st.plotly_chart(px.scatter(sample, x=vcol, y=scol, opacity=0.5, title=f"SHAP value vs. {feat}").update_layout(**F.LAYOUT))
                elif vcol in sample:
                    st.plotly_chart(px.box(sample, x=vcol, y=scol, title=f"SHAP value by {feat}").update_layout(**F.LAYOUT))
        if perm is not None:
            st.plotly_chart(F.importance_fig(perm, "importance_mean", "Permutation importance (drop in test ROC-AUC)"))
            st.caption("Permutation importance is model-agnostic and shown as a cross-check (or as the substitute when SHAP is unavailable).")

    with tab2:
        if not shap_available() or rep["method"] != "SHAP":
            st.info("Local explanations need SHAP (`pip install shap`) and a pipeline run that used it. Re-run `python -m src.pipeline --stages explain` after installing.")
        else:
            bundle = get_bundle(paths)
            meta = bundle["meta"]
            snaps = pget(paths.churn_snapshots)
            feats = meta["numeric_features"] + meta["categorical_features"]
            test = snaps[(snaps["group"] == "test") & snaps["cutoff"].isin(pd.to_datetime(meta["test_cutoffs"]))]
            row_input = st.session_state.get("last_input")
            source = st.radio("Avatar to explain", ["A random held-out test avatar", "Last input from the Prediction Lab"],
                              index=1 if row_input is not None else 0, horizontal=True)
            if source.startswith("A random"):
                seed = st.number_input("Random seed", 0, 10_000, 0)
                sampled = test.sample(n=1, random_state=int(seed))
                X_row = sampled[feats]
                st.caption(f"Avatar id {int(sampled['avatar'].iloc[0])}; actual outcome: {'churned' if sampled['churned'].iloc[0] else 'retained'}.")
            elif row_input is None:
                st.info("Make a prediction in the Prediction Lab first.")
                st.stop()
            else:
                X_row = row_input[feats]
            bg = snaps[snaps["group"] != "test"][feats].sample(n=min(500, len(snaps)), random_state=0)
            ex = ShapExplainer(bundle["pipeline"], bg, list(meta["categorical_features"]))
            loc = ex.explain_local(X_row.reset_index(drop=True))
            st.write(f"Baseline (average model output): **{loc['base_value']:.3f}**; model output for this avatar: **{loc['model_output']:.3f}**. Units: {loc['units']}.")
            t = loc["table"].head(12).iloc[::-1]
            fig = px.bar(t, x="contribution", y="feature", orientation="h", color=t["contribution"] > 0,
                         color_discrete_map={True: "#c0392b", False: "#2471a3"}, title="Feature contributions (red raises churn risk, blue lowers it)")
            st.plotly_chart(fig.update_layout(showlegend=False, **F.LAYOUT))
            st.dataframe(loc["table"], hide_index=True)
