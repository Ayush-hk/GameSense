"""Headline numbers and short insight sentences for the Game Lab pages.

Everything is computed from the result tables passed in. Nothing here is typed in by hand,
so the text always agrees with the data on the page.
"""
from __future__ import annotations

import re

import pandas as pd

NEED_LABELS = {
    "autonomy_frustration": "Wishing to do something else",
    "competence_frustration": "Disappointed with own performance",
    "relatedness_frustration": "Toxic interactions with others",
}


def _band_key(label: str) -> int:
    m = re.match(r"\d+", str(label))
    return int(m.group()) if m else 10_000


def order_bands(bands) -> list[str]:
    return sorted({str(b) for b in bands}, key=_band_key)


def compact(n: float) -> str:
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 10_000:
        return f"{n / 1000:.0f}k"
    if abs(n) >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{n:,.0f}"


# ------------------------------------------------------------------ games played / age

def played_kpis(playtime: pd.DataFrame, most: pd.DataFrame) -> list[dict[str, str]]:
    top = most.iloc[0] if len(most) else None
    return [
        {"label": "Participants with play data", "value": f"{playtime['pid'].nunique():,}", "sub": "Xbox, Nintendo and Steam", "icon": "👥"},
        {"label": "Named games ranked", "value": f"{len(most):,}", "sub": "played by 10+ people", "icon": "🎮", "tone": "green"},
        {"label": "Hours of play recorded", "value": compact(playtime["hours"].sum()), "sub": "all platforms", "icon": "⏱", "tone": "amber"},
        {"label": "Most played game", "value": str(top["game"]) if top is not None else "n/a",
         "sub": f"{compact(top['hours'])} hours" if top is not None else "", "icon": "🏆", "tone": "pink"},
    ]


def age_kpis(participants: pd.DataFrame, playtime: pd.DataFrame, genre_pop: pd.DataFrame) -> list[dict[str, str]]:
    per_person = playtime.groupby("pid")["hours"].sum()
    top_genre = genre_pop.iloc[0] if len(genre_pop) else None
    return [
        {"label": "Total participants", "value": f"{len(participants):,}", "sub": "qualified in the study", "icon": "👥"},
        {"label": "Age range", "value": f"{int(participants['age'].min())} - {int(participants['age'].max())}",
         "sub": f"Mean: {participants['age'].mean():.1f}", "icon": "🎂", "tone": "blue"},
        {"label": "Genres", "value": f"{len(genre_pop)}", "sub": f"Most played: {top_genre['genre']}" if top_genre is not None else "", "icon": "🧩", "tone": "green"},
        {"label": "Median hours per person", "value": f"{per_person.median():,.0f}" if len(per_person) else "n/a", "sub": "all platforms", "icon": "⏱", "tone": "pink"},
    ]


def age_genre_insights(matrix: pd.DataFrame, n: int = 5, min_players: int = 30) -> list[dict[str, str]]:
    """Genres with the strongest age skew: the band where each genre peaks and by how much (lift - 1).

    Only cells with at least min_players players are considered, so a handful of people cannot headline a genre.
    """
    rows = []
    big = matrix[matrix["players"] >= min_players] if "players" in matrix.columns else matrix
    for genre, g in big.groupby("genre"):
        g = g.sort_values("lift", ascending=False)
        best = g.iloc[0]
        if best["lift"] <= 1.0:          # no band spends more than average on this genre: nothing to highlight
            continue
        text = f"Highest lift in {best['age_band']} ({best['lift']:.2f})"
        if len(g) > 1 and g.iloc[1]["lift"] > 1:
            text += f" and {g.iloc[1]['age_band']} ({g.iloc[1]['lift']:.2f})"
        rows.append((float(best["lift"]), {"title": str(genre), "text": text, "pill": f"{(best['lift'] - 1) * 100:+.0f}%"}))
    rows.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in rows[:n]]


def age_key_insight(matrix: pd.DataFrame, min_players: int = 30) -> str:
    if "players" in matrix.columns:
        matrix = matrix[matrix["players"] >= min_players]
    bands = order_bands(matrix["age_band"])
    if len(bands) < 2:
        return "Not enough age bands to compare."
    lo, hi = bands[0], bands[-1]

    def top(band: str) -> str:
        g = matrix[(matrix["age_band"] == band) & (matrix["lift"] > 1)].sort_values("lift", ascending=False).head(2)
        return " and ".join(g["genre"]) if len(g) else "no genre in particular"
    return f"Players aged {lo} spend relatively more time in {top(lo)}, while {hi} players favour {top(hi)}."


# ------------------------------------------------------------------ frustration

def need_means(diary: pd.DataFrame) -> list[tuple[str, float]]:
    out = [(NEED_LABELS[c], float(diary[c].mean())) for c in NEED_LABELS if c in diary.columns]
    return sorted(out, key=lambda t: t[1], reverse=True)


def frustration_kpis(diary: pd.DataFrame, by_genre: pd.DataFrame, ranking: pd.DataFrame) -> list[dict[str, str]]:
    items = [{"label": "Average frustration (1-7)", "value": f"{diary['frustration'].mean():.2f}",
              "sub": f"{len(diary):,} diary entries", "icon": "😤"}]
    if len(by_genre):
        hi, lo = by_genre.iloc[0], by_genre.iloc[-1]
        items.append({"label": "Most frustrating genre", "value": str(hi["genre"]), "sub": f"Avg. {hi['shrunken_mean']:.2f}", "icon": "🔥", "tone": "pink"})
        items.append({"label": "Least frustrating genre", "value": str(lo["genre"]), "sub": f"Avg. {lo['shrunken_mean']:.2f}", "icon": "😌", "tone": "green"})
    if len(ranking):
        r = ranking.iloc[0]
        items.append({"label": "Top frustration title", "value": str(r.get("game", r["game_key"])), "sub": f"Avg. {r['shrunken_mean']:.2f}", "icon": "🎯", "tone": "amber"})
    return items


def frustration_quick_insights(by_genre: pd.DataFrame, by_platform: pd.DataFrame, by_age: pd.DataFrame,
                               needs: list[tuple[str, float]]) -> list[str]:
    out: list[str] = []
    if len(by_genre) >= 2:
        a, b = by_genre.iloc[0], by_genre.iloc[1]
        out.append(f"{a['genre']} and {b['genre']} have the highest frustration scores (avg. {(a['shrunken_mean'] + b['shrunken_mean']) / 2:.2f}).")
    if len(by_platform) >= 2:
        p = by_platform.sort_values("mean", ascending=False)

        def who(x: str) -> str:
            return "Players on other platforms" if x == "Other" else f"{x} players"
        out.append(f"{who(p.iloc[0]['platform'])} report the highest frustration (avg. {p.iloc[0]['mean']:.2f}); "
                   f"{who(p.iloc[-1]['platform']).replace('Players', 'players')} the lowest ({p.iloc[-1]['mean']:.2f}).")
    if len(by_age) >= 2:
        a = by_age.sort_values("mean", ascending=False).iloc[0]
        out.append(f"The {a['age_band']} age band reports the highest frustration (avg. {a['mean']:.2f}).")
    if needs:
        out.append(f"Highest-scoring item overall: {needs[0][0].lower()} (avg. {needs[0][1]:.2f}).")
    return out


# ------------------------------------------------------------------ models

def model_kpis(frus: pd.DataFrame, recs: pd.DataFrame, reception: pd.DataFrame, k: int = 10) -> list[dict[str, str]]:
    def auc(df: pd.DataFrame, name: str) -> str:
        r = df[df["model"] == name]
        return f"{r['roc_auc'].iloc[0]:.2f}" if len(r) else "n/a"
    hit = recs[(recs["k"] == k) & (recs["model"] == "item_item")]
    pop = recs[(recs["k"] == k) & (recs["model"] == "popularity")]
    return [
        {"label": "Frustration model (ROC-AUC)", "value": auc(frus, "logreg"), "sub": "held-out participants", "icon": "😤", "tone": "pink"},
        {"label": "Reception model (ROC-AUC)", "value": auc(reception, "forest"), "sub": "later releases", "icon": "⭐", "tone": "amber"},
        {"label": f"Recommender hit rate (top {k})", "value": f"{hit['hit_rate'].iloc[0]:.0%}" if len(hit) else "n/a",
         "sub": f"most popular: {pop['hit_rate'].iloc[0]:.0%}" if len(pop) else "", "icon": "🎮", "tone": "green"},
    ]
