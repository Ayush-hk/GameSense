# Limitations

## Sample selection (most important)
* Every avatar in the sample has 262-499 polls, i.e. about 44-83 hours online in three years. This condition looks at the avatar's
  whole life, so it removes avatars that quit after very little play and avatars that played heavily. A model trained here sees
  neither kind. Churn rates (about 33-36%) and scores should not be read as population values.
* One consequence: estimated hours online barely differs between segments (medians 55-60 hours), so segments are
  separated by *how* hours are spread (regularity, session length, level), not by how much was played.
* How the subset was drawn is undocumented, so the size of the bias cannot be measured here.

## Data
* Avatar is not player; one person can run several avatars.
* One realm, one faction, 2006-2009. No race/class/zone/guild in this file; no purchases, chat or social data.
* 10-minute polling: short logins are missed, session lengths are estimates, polling gaps can look like inactivity.
* Tenure is left-censored (an avatar first seen in Jan 2006 may be older).

## Churn model
* Churn means "no poll in the next 28 days" for an avatar active in the previous 28 days. It cannot separate quitting from a break, a
  switch of character or a logging gap. Windows are configurable.
* Test set: 417 rows from 236 avatars across 3 periods, so intervals are wide (ROC-AUC 0.72-0.81).
* The model is only modestly better than "days since last seen" alone; most of its skill is recency.
* The tuned threshold (0.41) favours recall (0.84) at the price of precision (0.55); choose a threshold for the use case.
* Probabilities are not calibrated.

## Segmentation
* Silhouette 0.20 is low: segments overlap and behaviour is a continuum. Segments describe differences; they are not types of player.
* Segments use each avatar's whole history, so they are descriptive and are not used as churn inputs.

## Explainability
* SHAP is not installed in the build environment; permutation importance was used instead and says so in the report. SHAP code
  exists but has not been run. Neither method shows causes; correlated features share credit.

## Engineering
* Streamlit, Plotly and SHAP could not be installed in the build environment, so the dashboard has been syntax-checked, never launched.
* The full 10.8M-row Kaggle file has not been processed.

# Game Lab

## Who is in the data
* Open Play participants are adult volunteers in the US and UK aged 18-43. Nothing here says anything about teenagers, children or people over 43.
* Only people who agreed to share platform data and filled in the diary are included (954 of 2,310 qualified participants have diary entries). They are probably more engaged than the average player.
* Age effects are cross-sectional: different people at different ages, not the same people over time. Correlations between age and genre share are small (about 0.05-0.14).

## Play time
* Hours come from Xbox, Nintendo and Steam telemetry. There is no PlayStation, mobile or most free-to-play telemetry in the tables used, so Fortnite, Minecraft, Roblox and mobile games are under-represented in "most played".
* Xbox titles have no names in the data, only a genre; 52% of all hours are in unnamed titles. Game rankings use named titles only.
* Genres come from three vocabularies merged by a hand-written map; some genres are coarse ("Adventure" covers a lot).

## Frustration
* It is self-report about the game the person played most recently, measured with three items (alpha 0.63, which is modest). It is not a diagnosis of anything.
* A person's ratings are strongly personal (ICC 0.47): some people rate everything higher. The "versus the player's own average" view removes this; the plain ranking does not.
* "High frustration" (3 or more) is a chosen cut-off, not validated.
* Rankings use games with at least 10 diarists and 20 entries (the same minimum as every other group on the pages); shrinkage reduces but does not remove noise. Intervals are wide (about +/-0.4).
* Diary names are free text. Variants were merged with an alias list that I wrote after looking at the data; some entries will still be split or wrongly merged.
* Competitive multiplayer games rank highest. The data cannot say whether the game design, losing, other players or the kind of person who plays it is the cause.
* Steam review share vs diary frustration: Spearman -0.66 (p = 0.003), but on only 18 games that appear in both. It is a correlation across games, nothing more.

## Models
* Frustration model: most of the skill is the player's own previous ratings; it needs a history to work. It does not say why.
* Reception model: Steam tags are voted by the community and keep changing after release, so this is not a strict pre-launch predictor. Release year is missing for 23% of the catalogue, including some very large older games, so those are not in it. Test games (2024) are more often positively reviewed (66%) than training games (52%), so the base rate shifted. Skill is modest (0.68).
* Recommender: the list of 302 games was chosen using all participants (games with at least 10 players), including the ones held out in the test, which slightly favours every model equally. Offline hit rate on real play hours of 1,480 people and 302 games. It has not been tried with real users, and "played at least 5 hours" is a rough sign of liking.
* Streamlit and Plotly were not installable in the build environment. The page code was executed against the real result files with stand-in modules (see the verification report); how the pages look and behave in a browser has not been seen.
