import pandas as pd
import streamlit as st

from src.dashboard import ui
from src.dashboard.common import get_bundle, jget, origin_banner, pget, require, setup
from src.errors import GameSenseError
from src.models.predict import predict_churn
from src.reporting import figures as F

cfg, paths = setup("Prediction Lab", badge="Churn project")
metrics = jget(paths.metrics)
require(metrics is not None and (paths.churn_model.is_file() or paths.churn_meta.is_file()),
        "No trained churn model yet. Place the real dataset in `data/raw/` and run `python -m src.pipeline`.")
origin_banner(paths)
definition = jget(paths.churn_definition)
bundle = get_bundle(paths)
meta = bundle["meta"]

sel = metrics["selected_model"]
tuned = metrics["test_selected_tuned_threshold"]
ci = metrics["test_bootstrap_ci"]
lo_auc, hi_auc = ci["models"][sel]["roc_auc"]
main, side = ui.layout()
with side:
    ui.text_panel("Key insight", [definition["definition"]], key=True)
    ui.panel("Model summary", [
        {"title": "Selected model", "text": f"{sel}; chosen on validation data"},
        {"title": "Decision threshold", "text": "Tuned on validation data", "pill": f"{tuned['threshold']:.3f}"},
        {"title": "ROC-AUC 95% interval", "text": f"{lo_auc:.3f} to {hi_auc:.3f} (resampling avatars)"},
        {"title": "Churn prevalence", "text": f"{tuned['n']:,} test rows from {ci['n_avatars']} avatars", "pill": f"{tuned['prevalence']:.1%}"},
    ])
with main:
    st.info(definition["definition"])

    tab1, tab2, tab3 = st.tabs(["Model evaluation", "Score an avatar", "Score a CSV"])

    with tab1:
        st.write(f"**Selected model:** `{sel}` - chosen on validation data; threshold {tuned['threshold']:.3f} tuned on validation. All figures below are on held-out avatars at the newest cutoff.")
        ui.kpis([{"label": lab, "value": "n/a" if tuned[key] is None else f"{tuned[key]:.3f}", "sub": "held-out avatars", "icon": ic, "tone": tn}
                 for (lab, key), ic, tn in zip([("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Precision", "precision"), ("Recall", "recall")],
                                                ["📈", "🎯", "✅", "🔎"], ["", "blue", "green", "pink"])])
        rec = metrics["test"]["recency_only"]["roc_auc"]
        ui.note(f"Reference points on the same test rows: prior-only ROC-AUC {metrics['test']['dummy_prior']['roc_auc']:.3f}; "
                f"'days since last seen' alone {rec:.3f}. F1 is {'n/a' if tuned['f1'] is None else format(tuned['f1'], '.3f')}.")
        comp = pd.read_csv(paths.model_comparison) if paths.model_comparison.is_file() else None
        if comp is not None:
            st.dataframe(comp, hide_index=True)
        a, b = st.columns(2)
        for col, fn in ((a, F.roc_fig), (b, F.pr_fig)):
            fig = fn(paths)
            if fig is not None:
                col.plotly_chart(fig)
        fig = F.confusion_fig(paths)
        if fig is not None:
            st.plotly_chart(fig)
        for w in metrics.get("warnings", []):
            st.warning(w)

    with tab2:
        st.caption("Enter the behaviour of one avatar over a 28-day observation window. Inputs are validated against plausible ranges.")
        rules = meta["feature_rules"]
        snaps = pget(paths.churn_snapshots)
        defaults = snaps[meta["numeric_features"]].median() if snaps is not None else pd.Series(dtype=float)
        vals = {}
        cols = st.columns(3)
        for i, f in enumerate(meta["numeric_features"]):
            lo, hi = rules.get(f, [0, None])
            default = float(defaults.get(f, 0.0))
            vals[f] = cols[i % 3].number_input(f, min_value=float(lo) if lo is not None else None, max_value=float(hi) if hi is not None else None,
                                               value=default, key=f"in_{f}")
        for f in meta["categorical_features"]:
            options = meta["categories_seen"][f]
            vals[f] = st.selectbox(f, options, key=f"in_{f}")
        if st.button("Predict churn risk"):
            try:
                res, warns = predict_churn(bundle, pd.DataFrame([vals]))
                p = float(res["churn_probability"].iloc[0])
                st.metric("Estimated churn probability", f"{p:.1%}")
                st.write("Predicted class:", "**churn**" if res["predicted_churn"].iloc[0] else "**retained**", f"(threshold {meta['threshold']:.3f})")
                for w in warns:
                    st.warning(w)
                st.session_state["last_input"] = pd.DataFrame([vals])
                st.caption("Open the Explainability page to see which features drove this prediction.")
            except GameSenseError as exc:
                st.error(str(exc))

    with tab3:
        st.caption(f"Upload a CSV with columns: {', '.join(meta['numeric_features'] + meta['categorical_features'])}")
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            try:
                df = pd.read_csv(up)
                res, warns = predict_churn(bundle, df)
                for w in warns:
                    st.warning(w)
                out = pd.concat([df.reset_index(drop=True), res], axis=1)
                st.dataframe(out.head(500))
                st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), "churn_predictions.csv", "text/csv")
            except GameSenseError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
