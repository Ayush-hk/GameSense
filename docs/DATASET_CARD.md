# Dataset card

## Data actually used
| Item | Value |
|---|---|
| File | `data/raw/subset-wow.csv` |
| Obtained from | https://github.com/koaning/wow-avatar-datasets (file `subset-wow.csv`), cloned 2026-10-26 |
| Underlying data | World of Warcraft Avatar History (WoWAH), Lee, Chen, Cheng, Lei, ACM MMSys 2011 |
| SHA-256 | `03bacbc9b152ee0709e273c18f926b891504d7b0c060d72ae661364773817e18` |
| Size | 33,290,219 bytes |
| Rows | 1,017,282 read; 253 duplicate (avatar, timestamp) rows removed; 1,017,029 used |
| Avatars | 2,815 |
| Period | 2006-01-01 00:00:07 to 2009-01-10 05:08:13 |
| Columns | `player_id` (anonymised avatar id), `level` (1-80), `datetime` (ISO, e.g. 2007-06-12T03:18:28.000) |
| Missing values | none in any column |
| Polls per avatar | min 262, median 352, max 499 |

All figures above were measured by the pipeline (`reports/evaluation/dataset_card_generated.md` has the generated version).

## Origin and provenance
Real telemetry. Per the original authors (https://web.cs.wpi.edu/~claypool/mmsys-dataset/2011/wow), a script polled the in-game
`who` command every 10 minutes on one Taiwan realm (Light's Hope, Horde faction) for 1,107 days: 91,065 avatars and
667,032 sessions in the full study. Avatar and guild names were replaced by consistent random integers. The authors
note logins shorter than a poll can be missed and that maintenance and client problems left gaps.

The GitHub sample is a subset: the full set holds 91,065 avatars; this file holds 2,815. How the subset was drawn is
not documented in the repository. The only visible pattern is that every avatar has between 262 and 499 polls.

## Licence
Not stated in the GitHub repository. The authors' page allows download for research purposes and asks for the paper to be
cited. Treat this as research/educational use, cite the paper, and check the terms yourself before publishing the data.

## Kaggle version (not used here)
https://www.kaggle.com/datasets/mylesoneill/warcraft-avatar-history. Kaggle was not reachable, so its file list, size and licence were
not checked. A third-party copy of the full data shows the columns `char, level, race, charclass, zone, guild, timestamp`
(about 10.8M rows). The loader accepts either layout.

## Why not other Kaggle datasets
Several "online gaming behaviour" datasets turned up in search. For none could I verify that the data is real telemetry,
one is tagged "synthetic", and their targets are engagement levels or retention flags with no timeline, so they cannot
support a churn label defined from inactivity.

## Preprocessing
Header matching ignores case and spacing; the timestamp format is detected and checked over the whole file; rows with
unusable ids or levels outside 1-80 are dropped and counted; duplicate (avatar, timestamp) rows are removed; data are sorted by
time. The raw file is never changed. A "session" is a run of polls with no gap over 30 minutes; session length is
the first-to-last poll span plus one poll interval.
