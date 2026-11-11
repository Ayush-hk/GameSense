# Architecture and workflow

```
data/raw/wowah_data.csv  (never modified)
        |  src/data/loader.py        chunked read, header/format detection, validation, cleaning log
        v
data/processed/events.pkl  --> daily/hourly aggregates, dataset_profile.json, dataset card
        |
        +--> src/features/behavior.py  window features (single engine, reads only [start, end))
        |        |
        |        +--> whole span ---> src/analytics/segmentation.py  (log1p + scale + KMeans, k by silhouette)
        |        |                     -> artifacts/preprocessors/, artifacts/models/, avatar_profiles.pkl
        |        |
        |        +--> per cutoff ---> src/features/churn_target.py   (obs window | prediction window, leakage checks)
        |                              -> src/models/churn.py        (Dummy / LogReg / RandomForest, validation -> refit -> test)
        |                              -> artifacts/models/churn_model.joblib, reports/evaluation/*
        v
src/explainability/explain.py  SHAP (if installed) else permutation importance
src/reporting/report.py        evaluation_report.md, insights.json (composed only from result files)
app.py + pages/*               Streamlit; reads result files, shows "not available" when absent
```

## Game Lab
```
data/raw_games/open_play/*.csv.gz      src/games/openplay.py   qualified participants, telemetry -> hours per game, diary -> frustration
data/raw_games/steam_insights/*.zip    src/games/steamcat.py   reviews, owners, price, genres, tags
        |                              src/games/names.py      game-name cleaning and aliases
        v
src/games/analysis.py   most played, age x genre (lift, trend with FDR), shrunken and within-person frustration, minimum-group rule
src/games/models.py     frustration model (grouped CV), Steam reception model (time split), item-item recommender (leave-one-out)
src/games/pipeline.py   writes reports/games/*.csv|json and data/processed/games/*.pkl|joblib
pages/6..10             Streamlit pages that read those files (src/dashboard/games_common.py)
```

## Churn protocol
* Cutoff C: features from [C-28d, C); label from [C, C+28d). Avatar must be active in the observation window.
* Test cutoffs = the three newest 28-day periods with complete follow-up. Training cutoffs start 28 days before the earliest test observation window and go back in 28-day steps (24 cutoffs; the newest 4 validate, the rest fit).
* Avatars split into disjoint train/valid/test groups: fit on train avatars at the older training cutoffs, select model + threshold on valid avatars at the 4 newest training cutoffs, refit on train+valid avatars at all training cutoffs, evaluate once on test avatars at the 3 test cutoffs (with bootstrap intervals over avatars).
* `verify_split_integrity` and `assert_no_leakage` run on every build; unit tests cover them.

## Commands
`python -m src.pipeline [--raw-file PATH] [--stages ingest,segment,churn,explain,report]`; `streamlit run app.py`; `python -m unittest discover -s tests -t .`

Processed tables are local pickles (`.pkl`) written and read by the same machine; do not load pickles from other people.


## UI layer (pages 11-15 and styling)

- `src/dashboard/ui.py` - CSS and layout helpers (hero, KPI cards, panels, bar lists); all data text is HTML-escaped.
- `src/dashboard/profile.py` - validated local profile (`data/user/profile.json`, atomic write).
- `src/dashboard/content.py` - Terms, Privacy, FAQ and per-page notes text.
- `src/games/insights.py` - computed insight sentences and KPIs from real results.
- `.streamlit/config.toml` - base theme.
