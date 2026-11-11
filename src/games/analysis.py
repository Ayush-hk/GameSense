"""Descriptive analysis for the Game Lab: what is played, by whom, and how frustrating it is.

Every table that could expose individuals is filtered with a minimum-participants rule
(``games.min_group_participants``); nothing here is shown per person.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- helpers

GENRE_MAP = {
    "Role-playing (RPG)": "RPG", "Role Playing": "RPG",
    "Platform": "Platformer",
    "Simulator": "Simulation",
    "Sport": "Sports",
    "Racing + Flying": "Racing",
    "Puzzle + Trivia": "Puzzle", "Quiz/Trivia": "Puzzle",
    "Card & Board Game": "Card & Board", "Card + Board": "Card & Board",
    "Action + Adventure": "Adventure",
    "Multi-Player Online Battle Arena": "MOBA",
    "Real Time Strategy (RTS)": "RTS", "Turn-based strategy (TBS)": "Turn-based strategy",
    "Family + Kids": "Family",
    "Hack and slash/Beat 'em up": "Beat 'em up",
}
DROP_GENRES = {"Other"}


def harmonise_genres(text: str) -> list[str]:
    """Open Play mixes IGDB, Nintendo and Xbox genre names; map them to one set and remove duplicates."""
    out: list[str] = []
    if not isinstance(text, str):          # NaN / None for games without genre data
        return out
    for g in text.split(","):
        g = GENRE_MAP.get(g.strip(), g.strip())
        if g and g not in DROP_GENRES and g not in out:
            out.append(g)
    return out


def split_genres(s: pd.Series) -> pd.Series:
    return s.fillna("").map(harmonise_genres)


def explode_genres(playtime: pd.DataFrame) -> pd.DataFrame:
    """One row per (participant, game, genre). Hours are repeated for every genre of the game."""
    d = playtime[playtime["genres_clean"].notna()].copy()
    d["genre"] = split_genres(d["genres_clean"])
    return d.explode("genre").dropna(subset=["genre"])


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ---------------------------------------------------------------- what is played

def most_played(playtime: pd.DataFrame, platform: str | None = None, min_players: int = 10) -> pd.DataFrame:
    """Named games ranked by total hours; also players, median hours per player and share of all named hours."""
    d = playtime[playtime["named"]]
    if platform and platform != "All":
        d = d[d["platform"] == platform]
    g = d.groupby("game_key").agg(game=("game_name", "first"), hours=("hours", "sum"), players=("pid", "nunique"),
                                  median_hours_per_player=("hours", "median"),
                                  genres=("genres_clean", "first")).reset_index()
    g = g[g["players"] >= min_players].sort_values("hours", ascending=False)
    g["share_of_named_hours"] = g["hours"] / d["hours"].sum()
    return g.reset_index(drop=True)


def genre_popularity(playtime: pd.DataFrame, platform: str | None = None) -> pd.DataFrame:
    d = explode_genres(playtime)
    if platform and platform != "All":
        d = d[d["platform"] == platform]
    g = d.groupby("genre").agg(hours=("hours", "sum"), players=("pid", "nunique")).reset_index()
    g["share_of_hours"] = g["hours"] / g["hours"].sum()
    return g.sort_values("hours", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- who plays what (age)

def _min_cell(df: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    return df[df[col] >= k]


def age_genre_matrix(playtime: pd.DataFrame, participants: pd.DataFrame, min_participants: int = 10) -> pd.DataFrame:
    """Share of each age band's genre-hours going to each genre (rows sum to 1 across genres of that band).

    Games with several genres count toward each of them, so shares describe attention per genre, not a partition.
    """
    d = explode_genres(playtime).merge(participants[["pid", "age_band"]], on="pid")
    n_by_band = participants.groupby("age_band")["pid"].nunique()
    tot = d.groupby("age_band")["hours"].sum()
    g = d.groupby(["age_band", "genre"]).agg(hours=("hours", "sum"), players=("pid", "nunique")).reset_index()
    g["share"] = g["hours"] / g["age_band"].map(tot)
    g["band_participants"] = g["age_band"].map(n_by_band)
    return _min_cell(g, "players", min_participants).reset_index(drop=True)


def age_genre_lift(matrix: pd.DataFrame) -> pd.DataFrame:
    """Lift = a band's share of a genre divided by the all-ages share. >1 means the band spends relatively more of its time there."""
    allshare = matrix.groupby("genre")["hours"].sum() / matrix["hours"].sum()
    m = matrix.copy()
    m["lift"] = m["share"] / m["genre"].map(allshare)
    return m


def age_trend_by_genre(playtime: pd.DataFrame, participants: pd.DataFrame, min_participants: int = 30) -> pd.DataFrame:
    """Does a participant's share of play time in a genre change with age? Spearman (age vs share), BH-adjusted."""
    d = explode_genres(playtime).groupby(["pid", "genre"])["hours"].sum().reset_index()
    tot = playtime[playtime["genres_clean"].notna()].groupby("pid")["hours"].sum()
    d["share"] = d["hours"] / d["pid"].map(tot)
    wide = d.pivot(index="pid", columns="genre", values="share").fillna(0.0)
    wide = wide.join(participants.set_index("pid")["age"], how="inner")
    rows = []
    for genre in wide.columns.drop("age"):
        if (wide[genre] > 0).sum() < min_participants:
            continue
        rho, p = stats.spearmanr(wide["age"], wide[genre])
        rows.append({"genre": genre, "rho": float(rho), "p": float(p), "n_players": int((wide[genre] > 0).sum()),
                     "n_total": int(len(wide))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_adj"] = bh_fdr(out["p"].to_numpy())
    out["significant"] = out["p_adj"] < 0.05
    return out.sort_values("rho").reset_index(drop=True)


def top_games_by_age(playtime: pd.DataFrame, participants: pd.DataFrame, band: str, n: int = 10,
                     min_participants: int = 10) -> pd.DataFrame:
    """Named games ranked by hours inside one age band, with lift against everyone else."""
    d = playtime[playtime["named"]].merge(participants[["pid", "age_band"]], on="pid")
    inb = d[d["age_band"] == band]
    g = inb.groupby("game_key").agg(game=("game_name", "first"), hours=("hours", "sum"), players=("pid", "nunique")).reset_index()
    g = g[g["players"] >= min_participants].copy()
    band_total, all_total = inb["hours"].sum(), d["hours"].sum()
    all_hours = d.groupby("game_key")["hours"].sum()
    g["band_share"] = g["hours"] / band_total
    g["all_share"] = g["game_key"].map(all_hours) / all_total
    g["lift"] = g["band_share"] / g["all_share"]
    return g.sort_values("hours", ascending=False).head(n).reset_index(drop=True)


# ---------------------------------------------------------------- frustration

def cronbach_alpha(items: pd.DataFrame) -> float:
    x = items.dropna()
    k = x.shape[1]
    if k < 2 or len(x) < 3:
        return float("nan")
    var_sum = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - var_sum / total_var)) if total_var > 0 else float("nan")


def shrunken_frustration(diary: pd.DataFrame, by: str = "game_key", min_participants: int = 8,
                         min_entries: int = 20) -> pd.DataFrame:
    """Mean frustration per game (or genre/platform) with empirical-Bayes shrinkage toward the overall mean.

    Raw means for games with few ratings are unstable. The shrunken mean is
        (n*mean + m*grand) / (n + m)
    where m = within-group variance / between-group variance (method of moments), so small groups move toward
    the overall mean. Each person's entries are averaged first so one heavy diarist cannot dominate a game.
    """
    per = diary.groupby([by, "pid"]).agg(f=("frustration", "mean"), n=("frustration", "size"),
                                         high=("high_frustration", "mean")).reset_index()
    grand = per["f"].mean()
    g = per.groupby(by).agg(participants=("pid", "nunique"), entries=("n", "sum"), mean=("f", "mean"),
                            sd=("f", "std"), high_rate=("high", "mean")).reset_index()
    g = g[(g["participants"] >= min_participants) & (g["entries"] >= min_entries)].copy()
    if g.empty:
        return g.assign(shrunken_mean=pd.Series(dtype=float), se=pd.Series(dtype=float), ci_low=pd.Series(dtype=float),
                        ci_high=pd.Series(dtype=float), grand_mean=pd.Series(dtype=float), shrinkage_m=pd.Series(dtype=float))
    within = (per[per[by].isin(g[by])]["f"] - per[per[by].isin(g[by])][by].map(g.set_index(by)["mean"])).var(ddof=1)
    between = max(g["mean"].var(ddof=1) - within * (1.0 / g["participants"]).mean(), 1e-6)
    m = within / between
    g["shrunken_mean"] = (g["participants"] * g["mean"] + m * grand) / (g["participants"] + m)
    g["se"] = g["sd"] / np.sqrt(g["participants"])
    g["ci_low"] = g["shrunken_mean"] - 1.96 * g["se"]
    g["ci_high"] = g["shrunken_mean"] + 1.96 * g["se"]
    g["grand_mean"] = grand
    g["shrinkage_m"] = m
    return g.sort_values("shrunken_mean", ascending=False).reset_index(drop=True)


def within_person_delta(diary: pd.DataFrame, min_participants: int = 8, min_entries: int = 20) -> pd.DataFrame:
    """Frustration relative to the same person's own average, per game.

    This removes the fact that some people just rate everything higher. A positive delta means players
    report more frustration in this game than they do on average across the games they rated.
    """
    d = diary.copy()
    d["person_mean"] = d.groupby("pid")["frustration"].transform("mean")
    d["n_person"] = d.groupby("pid")["frustration"].transform("size")
    d = d[d["n_person"] >= 3]
    d["delta"] = d["frustration"] - d["person_mean"]
    per = d.groupby(["game_key", "pid"]).agg(delta=("delta", "mean"), n=("delta", "size")).reset_index()
    g = per.groupby("game_key").agg(participants=("pid", "nunique"), entries=("n", "sum"), delta=("delta", "mean"),
                                    sd=("delta", "std")).reset_index()
    g = g[(g["participants"] >= min_participants) & (g["entries"] >= min_entries)].copy()
    g["se"] = g["sd"] / np.sqrt(g["participants"])
    return g.sort_values("delta", ascending=False).reset_index(drop=True)


def frustration_reliability(diary: pd.DataFrame, items: list[str]) -> dict[str, float]:
    """Alpha of the three frustration items, and how stable a person's mean rating is (ICC from one-way ANOVA)."""
    alpha = cronbach_alpha(diary[items])
    per = diary.groupby("pid")["frustration"]
    k = per.size()
    k = k[k >= 2]
    sub = diary[diary["pid"].isin(k.index)]
    grand = sub["frustration"].mean()
    ssb = (sub.groupby("pid")["frustration"].agg(["mean", "size"]).pipe(lambda t: (t["size"] * (t["mean"] - grand) ** 2).sum()))
    ssw = ((sub["frustration"] - sub.groupby("pid")["frustration"].transform("mean")) ** 2).sum()
    msb = ssb / max(len(k) - 1, 1)
    msw = ssw / max(len(sub) - len(k), 1)
    k0 = (len(sub) - (k ** 2).sum() / len(sub)) / max(len(k) - 1, 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw) if (msb + (k0 - 1) * msw) > 0 else float("nan")
    return {"cronbach_alpha": alpha, "icc_person": float(icc), "participants": int(len(k)), "entries": int(len(sub))}


def frustration_by_group(diary: pd.DataFrame, group: str, min_participants: int = 10) -> pd.DataFrame:
    per = diary.groupby([group, "pid"])["frustration"].mean().reset_index()
    g = per.groupby(group).agg(participants=("pid", "nunique"), mean=("frustration", "mean"), sd=("frustration", "std")).reset_index()
    g["se"] = g["sd"] / np.sqrt(g["participants"])
    return g[g["participants"] >= min_participants].reset_index(drop=True)


def frustration_by_genre(diary: pd.DataFrame, games: pd.DataFrame, min_participants: int = 10, min_entries: int = 20) -> pd.DataFrame:
    """Join diary entries to the harmonised genres of the telemetry game table, then shrink per genre."""
    gm = games[["game_key", "genres"]].dropna()
    d = diary.merge(gm, on="game_key", how="inner")
    d["genre"] = split_genres(d["genres"])
    d = d.explode("genre").dropna(subset=["genre"])
    return shrunken_frustration(d, by="genre", min_participants=min_participants, min_entries=min_entries)


def steam_reception_link(delta: pd.DataFrame, lookup: pd.DataFrame) -> dict[str, float]:
    """Do games players find frustrating also get a lower share of positive Steam reviews? (Spearman, per game.)"""
    j = delta.merge(lookup, on="game_key", how="inner").dropna(subset=["pct_positive", "delta"])
    if len(j) < 15:
        return {"n_games": int(len(j)), "rho": float("nan"), "p": float("nan")}
    rho, p = stats.spearmanr(j["delta"], j["pct_positive"])
    return {"n_games": int(len(j)), "rho": float(rho), "p": float(p)}


def frustration_by_age(diary: pd.DataFrame, min_participants: int = 10) -> pd.DataFrame:
    return frustration_by_group(diary, "age_band", min_participants)


def need_profile(diary: pd.DataFrame, game_keys: list[str]) -> pd.DataFrame:
    """Which kind of frustration drives a game's score: autonomy (bangs_2), competence (4) or relatedness (6)."""
    d = diary[diary["game_key"].isin(game_keys)]
    return d.groupby("game_key")[["autonomy_frustration", "competence_frustration", "relatedness_frustration"]].mean().reset_index()
