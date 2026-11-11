"""Models for the Game Lab. All evaluation is on real data and held out by participant or by time.

1. Frustration: will a diary entry be high-frustration? (grouped CV by participant)
2. Reception: will a Steam game have >= 80% positive reviews? (chronological split)
3. Recommender: item-item collaborative filtering on real play hours (leave-one-out hit@k by user fold)
"""
from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.games.analysis import split_genres


# ---------------------------------------------------------------- shared

def _safe_auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def cluster_bootstrap_auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int = 300, seed: int = 0) -> tuple[float, float]:
    """95% interval for ROC-AUC, resampling whole groups (participants) so repeat entries do not fake precision."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        if len(np.unique(y[ii])) == 2:
            out.append(roc_auc_score(y[ii], p[ii]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"), float("nan"))


# ---------------------------------------------------------------- 1. frustration model

def _person_history(diary: pd.DataFrame) -> pd.Series:
    """Mean frustration of the same person's EARLIER entries only (NaN for the first entry)."""
    d = diary.sort_values(["pid", "date"])
    csum = d.groupby("pid")["frustration"].cumsum() - d["frustration"]
    cnt = d.groupby("pid").cumcount()
    return (csum / cnt.replace(0, np.nan)).reindex(diary.index)


def _frustration_frame(diary: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    d = diary.copy()
    d["person_prior"] = _person_history(d)
    d["has_prior"] = d["person_prior"].notna().astype(int)
    gm = games[["game_key", "genres"]].dropna()
    d = d.merge(gm, on="game_key", how="left")
    d["genre_list"] = split_genres(d["genres"])
    d["genre_known"] = d["genres"].notna().astype(int)
    return d


def _design(d: pd.DataFrame, game_prior: pd.Series, grand: float, genres: list[str], platforms: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["age"] = d["age"].astype(float)
    X["woman"] = (d["gender_group"] == "Woman").astype(int)
    X["another_gender"] = (d["gender_group"] == "Another gender").astype(int)
    X["person_prior"] = d["person_prior"].fillna(grand)
    X["has_prior"] = d["has_prior"]
    X["game_prior"] = d["game_key"].map(game_prior).fillna(grand)
    X["genre_known"] = d["genre_known"]
    for p in platforms:
        X[f"plat_{p}"] = (d["platform"] == p).astype(int)
    for g in genres:
        X[f"genre_{g}"] = d["genre_list"].map(lambda xs, g=g: int(g in xs))
    return X


def _game_prior(train: pd.DataFrame, grand: float, m: float = 6.0) -> pd.Series:
    """Shrunken mean frustration per game from TRAIN participants only (one mean per person first)."""
    per = train.groupby(["game_key", "pid"])["frustration"].mean().reset_index()
    g = per.groupby("game_key")["frustration"].agg(["mean", "size"])
    return (g["size"] * g["mean"] + m * grand) / (g["size"] + m)


def frustration_model(diary: pd.DataFrame, games: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Grouped 5-fold CV. Returns metrics (with participant-bootstrap CIs), baselines and the fitted final model."""
    seed = cfg["games"]["random_state"]
    d = _frustration_frame(diary, games).reset_index(drop=True)
    y = d["high_frustration"].to_numpy()
    groups = d["pid"].to_numpy()
    genres = [g for g, _ in pd.Series([x for xs in d["genre_list"] for x in xs]).value_counts().head(15).items()]
    platforms = sorted(d["platform"].unique())
    oof = {k: np.full(len(d), np.nan) for k in ("prevalence", "person_only", "game_only", "logreg", "forest")}
    gkf = GroupKFold(n_splits=cfg["games"]["recommender"]["folds"])
    for tr, te in gkf.split(d, y, groups):
        train, test = d.iloc[tr], d.iloc[te]
        grand = train["frustration"].mean()
        gp = _game_prior(train, grand)
        Xtr, Xte = _design(train, gp, grand, genres, platforms), _design(test, gp, grand, genres, platforms)
        # the game prior must not see the row it describes: recompute it leave-person-out inside train
        Xtr["game_prior"] = _loo_game_prior(train, grand)
        oof["prevalence"][te] = y[tr].mean()
        oof["person_only"][te] = Xte["person_prior"]
        oof["game_only"][te] = Xte["game_prior"]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(Xtr, y[tr])
        oof["logreg"][te] = lr.predict_proba(Xte)[:, 1]
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=15, n_jobs=-1, random_state=seed).fit(Xtr, y[tr])
        oof["forest"][te] = rf.predict_proba(Xte)[:, 1]
    rows = []
    for name, p in oof.items():
        lo, hi = cluster_bootstrap_auc(y, p, groups, seed=seed) if name != "prevalence" else (float("nan"),) * 2
        rows.append({"model": name, "roc_auc": 0.5 if name == "prevalence" else _safe_auc(y, p), "ci_low": lo, "ci_high": hi,
                     "pr_auc": float(average_precision_score(y, p))})
    metrics = pd.DataFrame(rows)
    # final model on all data, for the dashboard
    grand = d["frustration"].mean()
    gp = _game_prior(d, grand)
    X = _design(d, gp, grand, genres, platforms)
    X["game_prior"] = _loo_game_prior(d, grand)
    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X, y)
    coefs = pd.Series(final[-1].coef_[0], index=X.columns).sort_values()
    return {"metrics": metrics, "prevalence": float(y.mean()), "n_entries": int(len(d)), "n_participants": int(d["pid"].nunique()),
            "coefficients": coefs.reset_index().rename(columns={"index": "feature", 0: "coef_per_sd"}),
            "model": final, "game_prior": gp, "grand": float(grand), "genres": genres, "platforms": platforms,
            "columns": list(X.columns)}


def _loo_game_prior(train: pd.DataFrame, grand: float, m: float = 6.0) -> pd.Series:
    """Per-row game prior computed WITHOUT the row's own participant (avoids target leakage into the feature)."""
    per = train.groupby(["game_key", "pid"])["frustration"].agg(["sum", "size"]).reset_index()
    # one mean per person: use per-person means so heavy diarists do not dominate
    per["pm"] = per["sum"] / per["size"]
    gsum = per.groupby("game_key")["pm"].transform("sum")
    gcnt = per.groupby("game_key")["pm"].transform("size")
    loo = (gsum - per["pm"] + m * grand) / (gcnt - 1 + m)
    key = per.set_index(["game_key", "pid"])["pm"].index
    lookup = pd.Series(loo.to_numpy(), index=key)
    idx = pd.MultiIndex.from_arrays([train["game_key"].to_numpy(), train["pid"].to_numpy()])
    return pd.Series(lookup.reindex(idx).to_numpy(), index=train.index)


def predict_frustration(bundle: dict[str, Any], row: dict[str, Any]) -> float:
    """Probability of a high-frustration session for a hypothetical player/game (dashboard 'what if')."""
    d = pd.DataFrame([{"age": row["age"], "gender_group": row["gender_group"], "platform": row["platform"],
                       "game_key": row.get("game_key", ""), "person_prior": row.get("person_prior", np.nan),
                       "has_prior": int(pd.notna(row.get("person_prior", np.nan))),
                       "genre_list": row.get("genres", []), "genre_known": int(bool(row.get("genres")))}])
    X = _design(d, bundle["game_prior"], bundle["grand"], bundle["genres"], bundle["platforms"])[bundle["columns"]]
    return float(bundle["model"].predict_proba(X)[:, 1][0])


# ---------------------------------------------------------------- 2. Steam reception model

def reception_model(cat: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Predict >=80% positive reviews from price, free flag, release year, genres and tags (store-page information; tags are community-voted
    and keep changing after release, so this is not a strict pre-launch predictor).

    Owners and review counts are left out on purpose: they are only known after release.
    Chronological split: train up to train_until, validate up to valid_until, test after.
    """
    rc = cfg["games"]["steam_reception"]
    d = cat[(cat["total"] >= rc["min_reviews"]) & cat["release_year"].notna() & cat["pct_positive"].notna()].copy()
    feat = ["price_usd", "is_free", "release_year"] + [c for c in d.columns if c.startswith(("genre_", "tag_"))]
    d["price_usd"] = d["price_usd"].fillna(0.0).clip(upper=100)
    y = (d["pct_positive"] >= rc["positive_threshold"]).astype(int)
    tr_end, va_end = pd.Timestamp(rc["train_until"]).year, pd.Timestamp(rc["valid_until"]).year
    # year-level split (release_date was reduced to year): train <= 2022, valid = 2023, test = 2024
    tr = d["release_year"] <= tr_end
    va = (d["release_year"] > tr_end) & (d["release_year"] <= va_end if va_end > tr_end else d["release_year"] <= tr_end + 1)
    te = d["release_year"] > max(va_end, tr_end + 1) if va_end > tr_end else d["release_year"] > tr_end + 1
    if te.sum() < 100:
        te = d["release_year"] > tr_end
        va = pd.Series(False, index=d.index)
    X = d[feat]
    models = {
        "majority": None,
        "price_only": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3)),
        "forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=cfg["games"]["random_state"]),
    }
    cols = {"price_only": ["price_usd", "is_free"]}
    rows, preds = [], {}
    for name, m in models.items():
        if m is None:
            p = np.full(te.sum(), y[tr].mean())
        else:
            c = cols.get(name, feat)
            m.fit(X.loc[tr, c], y[tr])
            p = m.predict_proba(X.loc[te, c])[:, 1]
        preds[name] = p
        yt = y[te].to_numpy()
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(300):
            ii = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[ii])) == 2:
                bs.append(roc_auc_score(yt[ii], p[ii]))
        rows.append({"model": name, "roc_auc": _safe_auc(yt, p),
                     "ci_low": float(np.percentile(bs, 2.5)) if bs else float("nan"),
                     "ci_high": float(np.percentile(bs, 97.5)) if bs else float("nan"),
                     "pr_auc": float(average_precision_score(yt, p))})
    forest = models["forest"]
    imp = pd.Series(forest.feature_importances_, index=feat).sort_values(ascending=False).head(15)
    return {"metrics": pd.DataFrame(rows), "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum()),
            "positive_rate_train": float(y[tr].mean()), "positive_rate_test": float(y[te].mean()),
            "importance": imp.reset_index().rename(columns={"index": "feature", 0: "importance"}),
            "model": forest, "features": feat}


# ---------------------------------------------------------------- 3. recommender

def liked_matrix(playtime: pd.DataFrame, min_hours: float, min_players: int, min_user_games: int):
    """Binary user x game matrix of named games a person spent at least min_hours on (all platforms combined)."""
    d = playtime[playtime["named"]].groupby(["pid", "game_key"]).agg(hours=("hours", "sum"), game=("game_name", "first")).reset_index()
    d = d[d["hours"] >= min_hours]
    cnt = d.groupby("game_key")["pid"].nunique()
    d = d[d["game_key"].isin(cnt[cnt >= min_players].index)]
    n = d.groupby("pid")["game_key"].nunique()
    d = d[d["pid"].isin(n[n >= min_user_games].index)]
    users = np.array(sorted(d["pid"].unique()))
    items = np.array(sorted(d["game_key"].unique()))
    ui = {u: i for i, u in enumerate(users)}
    ii = {g: i for i, g in enumerate(items)}
    M = np.zeros((len(users), len(items)), dtype=np.float32)
    M[d["pid"].map(ui).to_numpy(), d["game_key"].map(ii).to_numpy()] = 1.0
    names = d.groupby("game_key")["game"].first().reindex(items).to_numpy()
    return M, users, items, names


def item_similarity(M: np.ndarray, shrink: float = 5.0) -> np.ndarray:
    """Cosine item-item similarity with co-occurrence shrinkage (pairs seen together rarely get less weight)."""
    co = M.T @ M
    norm = np.sqrt(np.diag(co))
    sim = co / np.outer(norm, norm).clip(min=1e-9)
    sim = sim * (co / (co + shrink))
    np.fill_diagonal(sim, 0.0)
    return sim


def recommend_scores(sim: np.ndarray, liked_idx: list[int]) -> np.ndarray:
    s = sim[liked_idx].sum(axis=0) if liked_idx else np.zeros(sim.shape[0])
    s[liked_idx] = -np.inf
    return s


def evaluate_recommender(M: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """User-fold leave-one-out: hide one liked game per held-out user, rank all unseen games, check if it is in the top k.

    Popularity and the item-item model are both built only from training users plus the held-out users' remaining games
    for the item model's inputs (the hidden game is never used). Hit@k is averaged over held-out users and 3 random hides.
    """
    rc = cfg["games"]["recommender"]
    rng = np.random.default_rng(cfg["games"]["random_state"])
    k_list = [5, rc["k"], 20]
    users = np.arange(M.shape[0])
    folds = np.array_split(rng.permutation(users), rc["folds"])
    hits = {m: {k: [] for k in k_list} for m in ("popularity", "item_item", "random")}
    for f in folds:
        train = np.setdiff1d(users, f)
        sim = item_similarity(M[train])
        pop = M[train].sum(axis=0)
        for u in f:
            liked = np.flatnonzero(M[u])
            for _ in range(3):
                hide = int(rng.choice(liked))
                rest = [i for i in liked if i != hide]
                scores = {"item_item": recommend_scores(sim, rest),
                          "popularity": np.where(np.isin(np.arange(M.shape[1]), rest), -np.inf, pop.astype(float)),
                          "random": np.where(np.isin(np.arange(M.shape[1]), rest), -np.inf, rng.random(M.shape[1]))}
                for m, s in scores.items():
                    order = np.argsort(-s, kind="stable")
                    rank = int(np.flatnonzero(order == hide)[0])
                    for k in k_list:
                        hits[m][k].append(rank < k)
    rows = []
    for m in hits:
        for k in k_list:
            h = np.array(hits[m][k], dtype=float)
            rows.append({"model": m, "k": k, "hit_rate": float(h.mean()), "n_trials": int(len(h)),
                         "ci_low": float(h.mean() - 1.96 * h.std(ddof=1) / np.sqrt(len(h))),
                         "ci_high": float(h.mean() + 1.96 * h.std(ddof=1) / np.sqrt(len(h)))})
    return {"metrics": pd.DataFrame(rows), "n_users": int(M.shape[0]), "n_games": int(M.shape[1]),
            "density": float(M.mean())}


def build_recommender(playtime: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    rc = cfg["games"]["recommender"]
    M, users, items, names = liked_matrix(playtime, rc["min_hours_liked"], 10, rc["min_user_games"])
    ev = evaluate_recommender(M, cfg)
    sim = item_similarity(M)
    return {"sim": sim, "items": items, "names": names, "popularity": M.sum(axis=0), "evaluation": ev}


def recommend(rec: dict[str, Any], liked_keys: list[str], k: int = 10, frustration: pd.DataFrame | None = None,
              frustration_weight: float = 0.0) -> pd.DataFrame:
    """Top-k games for a list of liked game keys. With frustration_weight > 0 the score is lowered for games whose
    shrunken frustration is above average (a small, explicit trade-off; weight 0 gives the plain model)."""
    pos = {g: i for i, g in enumerate(rec["items"])}
    liked = [pos[g] for g in liked_keys if g in pos]
    s = recommend_scores(rec["sim"], liked)
    df = pd.DataFrame({"game_key": rec["items"], "game": rec["names"], "score": s, "players": rec["popularity"]})
    df = df[np.isfinite(df["score"])]
    if frustration is not None and frustration_weight > 0:
        fz = frustration.set_index("game_key")
        centered = (fz["shrunken_mean"] - fz["grand_mean"]).reindex(df["game_key"]).fillna(0.0).to_numpy()
        df["frustration_vs_avg"] = centered
        scale = df["score"].max() or 1.0
        df["score"] = df["score"] / scale - frustration_weight * centered
    return df.sort_values("score", ascending=False).head(k).reset_index(drop=True)


# --------------------------------------------------------------------------- portable recommender files
def save_recommender_portable(rec: dict[str, Any], proc: Path) -> None:
    """Version-independent copy of the recommender: plain arrays (.npz) + a small table and a json file."""
    import json

    from src.portable import save_table
    proc = Path(proc)
    np.savez_compressed(proc / "recommender.npz", sim=np.asarray(rec["sim"], dtype="float32"),
                        items=np.asarray(rec["items"], dtype=str), names=np.asarray([str(n) for n in rec["names"]], dtype=str),
                        popularity=np.asarray(rec["popularity"], dtype="float32"))
    save_table(rec["evaluation"]["metrics"], proc / "recommender_metrics.pkl")
    ev = rec["evaluation"]
    (proc / "recommender_eval.json").write_text(json.dumps({"n_users": int(ev["n_users"]), "n_games": int(ev["n_games"]),
                                                            "density": float(ev["density"])}), encoding="utf-8")


def load_recommender(proc: Path) -> dict[str, Any]:
    """Load the recommender from the portable files (works with any pandas / scikit-learn version)."""
    import json

    from src.portable import read_table
    proc = Path(proc)
    z = np.load(proc / "recommender.npz", allow_pickle=False)
    meta = json.loads((proc / "recommender_eval.json").read_text(encoding="utf-8"))
    meta["metrics"] = read_table(proc / "recommender_metrics.pkl")
    return {"sim": z["sim"], "items": z["items"], "names": z["names"].astype(object), "popularity": z["popularity"], "evaluation": meta}
