# Game Lab data

The Game Lab answers: what do people play, which age groups play what, which games are played most, and which games
leave players frustrated. It uses two real datasets and no generated data.

## 1. Open Play (people, play time, diary)
| Item | Value |
|---|---|
| Source | https://github.com/digital-wellbeing/open-play (tables under `data/clean`), also archived on Zenodo, DOI 10.5281/zenodo.17536656 |
| Version in the file header | 1.2.7 (from `CITATION.cff`) |
| Authors | Ballou, Foldes, Vuorre, Hakman, Magnusson and others; full list in `data/raw_games/open_play/CITATION.cff` |
| Licence | CC0 1.0 **with a re-identification clause**: you may not try to identify participants. See `data/raw_games/open_play/LICENSE` |
| What it is | US and UK adults who shared play data from Xbox, Nintendo Switch and Steam, and filled in surveys and a daily diary |
| Used here | 2,310 qualified participants aged 18-43 (mean 27.1); 11,764 diary entries from 954 of them; 60,517 participant-by-title play-time rows |
| Files and SHA-256 | see the list at the bottom |

How it is used
* Only participants flagged `qualified` in the intake survey are kept.
* **What people play and for how long** comes from console and Steam telemetry (minutes per title), not from self-report.
* **Age** is the participant's age at intake, grouped in bands set in `configs/config.yaml` (18-22, 23-26, 27-30, 31-35, 36+).
* **Frustration** comes from the daily diary. On days a person played, they rated 1-7 how much they agreed with six
  Basic Needs in Games statements about the game they played most recently. Three are frustration items
  (`bangs_2` "I often found myself wishing I could do something else", `bangs_4` "I felt disappointed with my performance",
  `bangs_6` "Interactions with other players and/or characters felt toxic to me"); their mean is the frustration score. The other three
  (`bangs_1` played the way I wanted, `bangs_3` felt I made progress, `bangs_5` formed relationships) are satisfaction items, kept
  but not used for the ranking. Wording is from `codebook.xlsx` (sheet "Daily"). "High frustration" is a score of 3 or more (a setting, not a clinical cut-off).
* The diary names the game as free text. Names are cleaned (`src/games/names.py`) and joined to telemetry titles; 56% of diary
  entries match a telemetry title. The rest are mostly games with no shared telemetry (Fortnite, Minecraft, Roblox, mobile games).
  Frustration rankings use the diary names directly, so they do not depend on this match.
* Xbox telemetry has no game names in the data, only a genre, so Xbox contributes to genre results but not to game rankings.
* The three genre vocabularies in the data (IGDB, Nintendo, Xbox) are merged with a small hand-written map in
  `src/games/analysis.py` (for example "Platform" and "Platformer" become one genre).

Privacy: results are shown for groups only, and a group needs at least 10 participants (`min_group_participants`). No page
lists or looks up individuals.

## 2. Steam Insights catalogue (what the games are like)
| Item | Value |
|---|---|
| Source | Steam Catalog Insights snapshot (README title: October 2024; files dated 2024-12-10), exported from the Steam store and SteamSpy |
| Used | games, reviews, steamspy_insights, genres, tags (122,191 games after keeping `type = game`) |
| Licence | **Not found in the files I copied.** Check the licence of the repository you download it from before redistributing |
| Quirks | Quotes are backslash-escaped and missing values are `\N`; the loader handles both. About 23% of games have no release date |

It is used for the reception model (predict whether a game gets at least 80% positive reviews from price, release year,
genres and tags) and for one cross-check: do games players find frustrating also have a lower share of positive Steam reviews.

## Where to put the files
```
data/raw_games/open_play/        survey_intake.csv.gz, survey_daily.csv.gz, steam.csv.gz, nintendo.csv.gz,
                                 xbox.csv.gz, game_metadata.csv.gz   (+ LICENSE, CITATION.cff)
data/raw_games/steam_insights/   games.zip, reviews.zip, steamspy_insights.zip, genres.zip, tags.zip
```
Then `python -m src.games.pipeline`.

## Citation
Ballou, N., Foldes, T. A., Vuorre, M., Hakman, T., Magnusson, K., et al. Open Play: A longitudinal dataset of multi-platform video
game digital trace data and psychological measures. Zenodo, https://doi.org/10.5281/zenodo.17536656 (full author list in CITATION.cff).

## SHA-256 of the files used
```
200cd39225080f5ea35041da1651a4b3a128c5e15bc2010c1ad71fb0607bd4fc  open_play/game_metadata.csv.gz
941738a760d86a7c61c97ba408f912feb7aec99b0b4f892994b32c8149c77424  open_play/nintendo.csv.gz
1e54f473aa2c7863617d0348f9eec44c2c0814d201b6add9c7f1a4e1abecd8ce  open_play/steam.csv.gz
7e0a4060fe55e2e808185aacac8361fa129f35dc97e4156df8f88011ce5057f2  open_play/survey_daily.csv.gz
294ee5598e928805626b78b4eb4c2e0905ba3cfe128c6d4fcd5b28dcf228a485  open_play/survey_intake.csv.gz
f1f565e1715a71e8269f987b6b3a9354f8e205dbe3285aee00cd7e9e72496caa  open_play/xbox.csv.gz
bb86056960cd4f294fd74e023a89d798af58da53b2298660ad73f55a351158c2  steam_insights/games.zip
490ac3316f281ecbda3e57d0aa6303af42dd04ea8d0fc518d15d8f9b630afc55  steam_insights/reviews.zip
5a0108885960abb39411f4d3099edc6d62fe3674ae93f2a5f1eda745142ecb63  steam_insights/steamspy_insights.zip
a676a40a322449bb7325b75dc4590d9dec2d40fbc61fafb36732844ad21d68ea  steam_insights/genres.zip
8b084a0533d2517f87a5eb5f5998dd6002f0aae8230ebabee3ece21598645ba4  steam_insights/tags.zip
```
