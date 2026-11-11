# GameSense evaluation report

*Generated 2026-10-26T07:13:02+00:00 - data origin: `user_provided_file`*

## 1. Data

1,017,029 polls, 2,815 avatars, 2006-01-01 00:00:07 to 2009-01-10 05:08:13.

Source file `subset-wow.csv`, SHA-256 `03bacbc9b152ee0709e273c18f926b891504d7b0c060d72ae661364773817e18`.

## 2. Churn definition

Churn = zero polls in the 28-day prediction window following a cutoff, for avatars active (>= 1 active day) in the preceding 28-day observation window.

* Usable data: 2006-01-02 -> 2009-01-10
* Test cutoffs (newest first): 2008-12-13, 2008-11-15, 2008-10-18
* Training cutoffs (newest first; the newest 4 are used for validation, the rest for fitting): 2008-08-23 back to 2006-11-18 (24 cutoffs, 28 days apart)
* Training labels are fully known before the earliest test observation window begins.
* Avatars are split into disjoint train / validation / test groups; the test set is also the newest period (out-of-time).

| set | rows | churn rate |
|---|---|---|
| fit | 6601 | 0.320 |
| valid | 283 | 0.357 |
| refit | 9645 | 0.327 |
| test | 417 | 0.365 |

## 3. Model comparison

| split | model | ROC-AUC | PR-AUC | precision | recall | F1 |
|---|---|---|---|---|---|---|
| validation | dummy_prior | 0.500 | 0.357 | 0.000 | 0.000 | 0.000 |
| validation | recency_only | 0.697 | 0.507 | 0.521 | 0.624 | 0.568 |
| validation | logistic_regression | 0.751 | 0.573 | 0.510 | 0.792 | 0.620 |
| validation | random_forest | 0.778 | 0.616 | 0.519 | 0.802 | 0.630 |
| test | dummy_prior | 0.500 | 0.365 | 0.000 | 0.000 | 0.000 |
| test | recency_only | 0.729 | 0.556 | 0.572 | 0.625 | 0.597 |
| test | logistic_regression | 0.759 | 0.587 | 0.553 | 0.789 | 0.650 |
| test | random_forest | 0.772 | 0.577 | 0.571 | 0.743 | 0.646 |

**Selected model:** `random_forest` (highest validation ROC-AUC among Logistic Regression and Random Forest (ties: PR-AUC)).
**Threshold:** 0.410 (threshold maximising F1 on the validation set).

Test metrics at that threshold: precision 0.545, recall 0.836, F1 0.660, ROC-AUC 0.772, PR-AUC 0.577.

**95% bootstrap intervals on the test set (resampling avatars, 500 resamples, 236 avatars):**

| model | ROC-AUC 95% CI | PR-AUC 95% CI | ROC-AUC difference vs recency_only |
|---|---|---|---|
| dummy_prior | 0.500 to 0.500 | 0.318 to 0.412 | -0.272 to -0.175 |
| recency_only | 0.675 to 0.772 | 0.479 to 0.640 | - |
| logistic_regression | 0.708 to 0.802 | 0.506 to 0.665 | +0.002 to +0.057 |
| random_forest | 0.723 to 0.813 | 0.507 to 0.655 | +0.003 to +0.083 |

Confusion matrix (test, rows = actual):

| actual | predicted retained | predicted churned |
|---|---|---|
| retained | 159 | 106 |
| churned | 25 | 127 |

Class imbalance is handled with class-weighted training; because accuracy is misleading when classes are unbalanced, PR-AUC, precision and recall are reported alongside ROC-AUC.

**Warnings recorded during training**

* Metrics describe held-out avatars over 3 out-of-time period(s) on one game server; they are not guarantees for other games, servers or periods.
* Churn here means 'no activity in the prediction window' for an avatar, which is not the same as a player leaving the game.

## 4. Segmentation

k = 4: k=4 has the highest silhouette (0.203) among the values of k whose smallest cluster holds at least 2% of avatars.

Final silhouette (sample) 0.203; mean ARI across re-fits with other seeds 0.987.

2,815 avatars clustered; 0 excluded for having fewer than 3 active days.

| segment | avatars | description |
|---|---|---|
| 0 | 608 | higher: activity regularity (active days / lifetime span) (+1.4 SD), sessions per active day (+0.8 SD), mean session length (min) (+0.8 SD); lower: active days (-1.0 SD) |
| 1 | 1687 | lower: activity regularity (active days / lifetime span) (-0.5 SD) |
| 2 | 195 | higher: active days (+2.1 SD); lower: mean session length (min) (-2.2 SD), levels gained in window (-1.5 SD), level at end of window (-1.3 SD) |
| 3 | 325 | higher: level at end of window (+2.2 SD); lower: levels gained in window (-1.5 SD) |

## 5. Explainability

Method: **permutation importance (SHAP substitute)**. SHAP was not used because the 'shap' package is not installed in this environment; permutation importance (drop in test ROC-AUC) is shown instead.

Top features: days_since_last_active, level_last, n_sessions, active_days_last7, active_days, est_hours_online, sessions_per_active_day, level_gain.

> Feature attributions explain how the trained model produces its predictions. They are associations learned from observational data, not proof that changing a behaviour would cause an avatar to stay or leave.

## 6. Limitations

See `docs/LIMITATIONS.md`. Key points: avatar is not player; one server and one faction; polling gaps and maintenance windows; left-censored tenure; churn = inactivity, not departure; results are for this dataset and period only.
