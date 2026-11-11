# GameSense

Player behaviour analytics, segmentation and churn prediction for an MMORPG, built on real World of Warcraft
telemetry (the WoWAH "Avatar History" data collected by Lee, Chen, Cheng and Lei, 2006-2009).

The project reads a CSV of 10-minute server polls, turns it into per-avatar behaviour features, groups avatars
with K-Means, predicts who will be inactive over the next 28 days, and shows everything in a Streamlit dashboard.

## What was run on real data

Data used: `subset-wow.csv` from https://github.com/koaning/wow-avatar-datasets (a 1,017,282-row sample of the
original WoWAH data, 2,815 avatars, Jan 2006 - Jan 2009). The Kaggle copy
(https://www.kaggle.com/datasets/mylesoneill/warcraft-avatar-history) could not be reached from the build
environment, but the pipeline also accepts that file (see below).

| Result (test = unseen avatars, 3 newest 28-day periods) | Value |
|---|---|
| Churn model | Random Forest |
| ROC-AUC | 0.772 (95% CI 0.723 - 0.813) |
| PR-AUC | 0.577 (churn rate in test: 36.5%) |
| Precision / recall / F1 at tuned threshold 0.41 | 0.545 / 0.836 / 0.660 |
| "Days since last seen" alone (baseline) | ROC-AUC 0.729 |
| Logistic Regression | ROC-AUC 0.759 |
| Segments | k = 4, silhouette 0.203, seed stability (ARI) 0.987 |

The forest beats the one-feature baseline, but only by a small margin (difference in ROC-AUC between +0.003 and +0.083,
95% bootstrap). Recency is by far the strongest signal. Details: `reports/evaluation/evaluation_report.md`.

## Game Lab: what people play, by age, and what frustrates them
A second part of the project uses real play data from the Open Play study (US/UK adults aged 18-43 who shared Xbox,
Nintendo and Steam play time plus a daily diary) and a Steam catalogue snapshot. Data details, licence and citation:
`docs/GAMES_DATA.md`.

```powershell
python -m src.games.pipeline    # about 50 seconds; writes reports\games\ and data\processed\games\
streamlit run app.py            # pages 6-10 in the sidebar
```

| Page | What you can do |
|---|---|
| 6 What people play | Filter by platform, age band and genre; most played games by hours or by players; genre shares |
| 7 Age and games | Genre mix and lift per age band, top games in a band, which genres change with age |
| 8 Frustration Lab | Rank games by frustration (with intervals), compare genre / platform / age, see which kind of frustration drives it |
| 9 Game recommender | Pick games you like, get similar games, optionally avoid frustrating ones |
| 10 Game models | Held-out results for every model next to its baseline |

Results on the real data (numbers come from `reports/games/`):
* Most played named titles by total hours in the sample: Animal Crossing: New Horizons, Marvel Rivals, Zelda: Tears of the Kingdom,
  Pokemon Scarlet/Violet, Counter-Strike 2. Shooter, Adventure and RPG are the biggest genres by hours.
* Age: younger players put more of their time into platformers, shooters and tactical games; older players into RPG, adventure and
  family games. The effects are small (rank correlations about 0.05-0.14).
* Most frustrating games (shrunken mean on a 1-7 scale, at least 10 diarists, 48 games ranked): Dead by Daylight 3.17, Valorant 3.15,
  Counter-Strike 2 3.13, League of Legends 3.11, Overwatch 2.97. Least: Animal Crossing 1.79, Baldur's Gate 3 1.88, The Sims 4 1.88.
  Average frustration falls with age band (2.48 for 18-22, about 2.06 for 31+); person-level Spearman -0.17 (p < 0.001), still
  -0.11 after removing each game's average, so it is not only a difference in which games people play.
* Frustration model (will a diary entry be high-frustration; 954 people, participant-grouped CV): ROC-AUC 0.81 (0.79-0.82).
  A player's own earlier ratings alone give 0.80, the game alone 0.62, so the game adds a little.
* Steam reception model (80% positive or more, trained on releases up to 2022, tested on 2024 releases): ROC-AUC 0.68 (0.66-0.70)
  against 0.50 for the majority guess. Modest.
* Recommender (hide one liked game, is it in the top 10?): 32% against 20% for "most popular" and 4% for random.

## Setup (Windows PowerShell)
```powershell
cd GameSense
py -3.12 -m venv .venv          # any Python 3.10+ should work; 3.13 is what the tests ran on
.\.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Optional: `pip install -r requirements-optional.txt` adds SHAP (explanations) and pytest. The project works without them.

## Run
```powershell
python -m src.pipeline          # about 20 seconds on the included subset; rebuilds every model and report
streamlit run app.py            # dashboard at http://localhost:8501
python -m unittest discover -s tests -t .      # 91 tests (pytest also works)
```
Re-running the pipeline on your own machine is recommended: the saved `.joblib` models were written with
scikit-learn 1.9.1 and may not load under a different version.

## Using the full Kaggle file instead
Download `wowah_data.csv` from the Kaggle page above, delete `data\raw\subset-wow.csv`, put the Kaggle file in
`data\raw\`, then run `python -m src.pipeline`. It has race, class, zone and guild, and the code uses them
automatically (more features, class mix charts). The full file is ~10.8M rows; loading is chunked but
that size has not been tried here. If the timestamp format is not recognised, the run stops and says which
setting in `configs\config.yaml` to change.

## Folders
```
app.py, pages/        Streamlit dashboard (app.py builds the sidebar navigation; assets/logo.svg is the logo; (pages 1-5 churn project, 6-10 Game Lab,
                      11-15 Settings, Account & Profile, Terms, Privacy, Help & FAQ; 0 = Overview)
src/games             Game Lab: Open Play loader, Steam catalogue, analysis, models, pipeline
src/data              loading, validation, cleaning log
src/features          sessions, per-avatar features, churn labels and splits
src/analytics         descriptive stats, K-Means segmentation
src/models            training, evaluation, prediction input checks
src/explainability    SHAP (if installed) or permutation importance
src/reporting         report text and figures
configs/config.yaml   every analysis setting (windows, features, k range)
docs/                 dataset card, architecture, limitations, verification report
reports/              generated metrics, report and PNG figures
```

## Settings, profile and policy pages

Pages 11-15 are Settings (accent colour, default filters, notes on/off), Account & Profile,
Terms & Conditions, Privacy Policy and Help & FAQ. The profile is saved only on your own
machine in `data/user/profile.json` (no account or password). To show your name and contact
on the Terms and Help pages, fill `app.author` and `app.contact` in `configs/config.yaml`.
The interface is dark-only; `.streamlit/config.toml` holds the theme.

## Starting the app

From the project folder (the one that contains `app.py`):  `streamlit run app.py`

Saved tables are stored both as pickles and as a version-independent gzipped CSV (`*.csv.gz` + `*.schema.json`), and the
recommender as `recommender.npz`, so the dashboard also works with a different pandas / scikit-learn than the one that ran the
pipelines. If the saved churn model cannot be read by the installed scikit-learn, it is rebuilt from the saved training data
(a few seconds, once). `python tools/make_portable.py` creates the portable copies for data you produced yourself.
`run_gamesense.bat` / `start.py` are optional launchers that set up a private Python environment.

## Read before citing
This sample only has avatars with 262 to 499 polls (44 to 83 hours online) over three years. That filter is applied
to a whole life-time of play, so avatars that quit quickly or played extremely heavily are missing. Churn rates and
model scores here describe this sample, not all players. See `docs/LIMITATIONS.md`.
