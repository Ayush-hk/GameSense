"""Behavioural player segmentation (K-Means on scaled, log-stabilised features).

Method
------
1. Keep avatars with at least ``min_active_days`` active days (one-off avatars produce noise).
2. log1p the right-skewed features, then standardise every feature (StandardScaler).
3. Fit K-Means for each k in ``k_range`` and record inertia, silhouette (on a fixed-size
   random sample, for speed), Davies-Bouldin and the smallest cluster share.
4. Choose k = highest silhouette among the k whose smallest cluster is >= ``min_cluster_share``
   (falls back to the plain silhouette maximum, and says so, if none qualifies).
5. Check stability: re-fit with other seeds and report the mean Adjusted Rand Index.
6. Describe each cluster ONLY by how its standardised means differ from the overall
   population. No marketing names ("whales", "loyal") are attached.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.errors import EmptyDatasetError
from src.features.behavior import FEATURE_LABELS

UNCLUSTERED = -1


def make_preprocessor(features: list[str], log_features: list[str]) -> Pipeline:
    """log1p on skewed features -> StandardScaler on everything."""
    log_cols = [f for f in features if f in log_features]
    raw_cols = [f for f in features if f not in log_features]
    ct = ColumnTransformer(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), log_cols),
            ("raw", "passthrough", raw_cols),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("transform", ct), ("scale", StandardScaler())])


def _check_nonnegative(df: pd.DataFrame, cols: list[str]) -> None:
    neg = [c for c in cols if (df[c] < 0).any()]
    if neg:
        raise ValueError(f"Segmentation features must be non-negative for log scaling: {neg}")


def select_k(
    Xs: np.ndarray, k_range: tuple[int, int], min_share: float, sil_sample: int, seed: int
) -> tuple[pd.DataFrame, int, str]:
    rows = []
    rng = np.random.RandomState(seed)
    n = Xs.shape[0]
    sample_idx = rng.choice(n, size=min(sil_sample, n), replace=False)
    for k in range(int(k_range[0]), int(k_range[1]) + 1):
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(Xs)
        shares = np.bincount(labels, minlength=k) / n
        sub = labels[sample_idx]
        sil = float(silhouette_score(Xs[sample_idx], sub)) if len(np.unique(sub)) > 1 else float("nan")
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": sil,
                "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
                "min_cluster_share": float(shares.min()),
                "max_cluster_share": float(shares.max()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise EmptyDatasetError("Not enough avatars to fit any clustering.")
    ok = table[table["min_cluster_share"] >= min_share]
    if len(ok):
        best = ok.loc[ok["silhouette"].idxmax()]
        reason = (
            f"k={int(best['k'])} has the highest silhouette ({best['silhouette']:.3f}) among the values of k "
            f"whose smallest cluster holds at least {min_share:.0%} of avatars."
        )
    else:
        best = table.loc[table["silhouette"].idxmax()]
        reason = (
            f"No k satisfied the minimum-cluster-share rule ({min_share:.0%}); "
            f"k={int(best['k'])} was chosen purely by highest silhouette ({best['silhouette']:.3f})."
        )
    return table, int(best["k"]), reason


def describe_clusters(z_means: pd.DataFrame, threshold: float = 0.5, top: int = 3) -> dict[int, str]:
    """Plain-language descriptors derived only from standardised feature differences."""
    out: dict[int, str] = {}
    for cl, row in z_means.iterrows():
        higher = row[row >= threshold].sort_values(ascending=False).head(top)
        lower = row[row <= -threshold].sort_values().head(top)
        parts = []
        if len(higher):
            parts.append("higher: " + ", ".join(f"{FEATURE_LABELS.get(f, f).lower()} ({v:+.1f} SD)" for f, v in higher.items()))
        if len(lower):
            parts.append("lower: " + ", ".join(f"{FEATURE_LABELS.get(f, f).lower()} ({v:+.1f} SD)" for f, v in lower.items()))
        out[int(cl)] = "; ".join(parts) if parts else "close to the population average on every feature"
    return out


def fit_segmentation(profiles: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit the full segmentation. Returns fitted objects plus all measured results."""
    sc = cfg["segmentation"]
    features: list[str] = [f for f in sc["features"] if f in profiles.columns]   # skip features the data cannot provide
    dropped = [f for f in sc["features"] if f not in profiles.columns]
    seed = int(sc["random_state"])
    eligible = profiles[profiles["active_days"] >= sc["min_active_days"]].copy()
    if len(eligible) < 50:
        raise EmptyDatasetError(
            f"Only {len(eligible)} avatars have >= {sc['min_active_days']} active days; need at least 50 to cluster."
        )
    _check_nonnegative(eligible, [f for f in features if f in sc["log_features"]])

    pre = make_preprocessor(features, list(sc["log_features"]))
    Xs = pre.fit_transform(eligible[features])
    selection, best_k, reason = select_k(
        Xs, tuple(sc["k_range"]), float(sc["min_cluster_share"]), int(sc["silhouette_sample"]), seed
    )
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(Xs)
    labels = km.labels_

    # ---- stability across seeds (on a sample, ARI vs. the final labelling)
    rng = np.random.RandomState(seed)
    samp = rng.choice(len(Xs), size=min(20000, len(Xs)), replace=False)
    aris = []
    for s in range(1, int(sc["stability_seeds"]) + 1):
        other = KMeans(n_clusters=best_k, n_init=5, random_state=seed + s).fit(Xs[samp])
        aris.append(float(adjusted_rand_score(labels[samp], other.labels_)))

    # ---- final silhouette on sample
    sil_idx = rng.choice(len(Xs), size=min(int(sc["silhouette_sample"]), len(Xs)), replace=False)
    final_sil = float(silhouette_score(Xs[sil_idx], labels[sil_idx])) if best_k > 1 else float("nan")

    # ---- profiles
    names = list(pre.get_feature_names_out())
    z = pd.DataFrame(Xs, columns=names)
    z["cluster"] = labels
    z_means = z.groupby("cluster")[names].mean()
    z_means = z_means.rename(columns=lambda c: c)  # names already match features
    eligible["segment"] = labels
    prof_rows = []
    descriptors = describe_clusters(z_means[[f for f in features]])
    for cl, grp in eligible.groupby("segment"):
        row = {"segment": int(cl), "avatars": int(len(grp)), "share": float(len(grp) / len(eligible))}
        for f in features:
            row[f"median_{f}"] = float(grp[f].median())
        if "charclass" in grp.columns and grp["charclass"].notna().any():
            row["modal_class"] = str(grp["charclass"].mode().iloc[0])
        if "has_guild" in grp.columns:
            row["share_in_guild_at_end"] = float(grp["has_guild"].mean())
        row["description"] = descriptors[int(cl)]
        prof_rows.append(row)
    cluster_profiles = pd.DataFrame(prof_rows).sort_values("segment").reset_index(drop=True)

    # ---- 2-D projection for plotting
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    proj = pca.transform(Xs)
    eligible["pc1"], eligible["pc2"] = proj[:, 0], proj[:, 1]

    out_profiles = profiles.copy()
    elig_key = eligible["avatar"].to_numpy()
    for col in ("segment", "pc1", "pc2"):
        out_profiles[col] = out_profiles["avatar"].map(pd.Series(eligible[col].to_numpy(), index=elig_key))
    out_profiles["segment"] = out_profiles["segment"].fillna(UNCLUSTERED).astype(int)

    report = {
        "features": features,
        "features_unavailable_in_data": dropped,
        "log_features": [f for f in features if f in sc["log_features"]],
        "min_active_days": int(sc["min_active_days"]),
        "avatars_total": int(len(profiles)),
        "avatars_clustered": int(len(eligible)),
        "avatars_excluded_insufficient_activity": int(len(profiles) - len(eligible)),
        "chosen_k": best_k,
        "k_selection_reason": reason,
        "silhouette_final_sample": final_sil,
        "stability_ari_mean": float(np.mean(aris)) if aris else None,
        "stability_ari_each": aris,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cluster_sizes": {int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "descriptors": {int(k): v for k, v in descriptors.items()},
        "z_means": z_means[features].round(4).to_dict(orient="index"),
    }
    return {
        "preprocessor": pre,
        "kmeans": km,
        "selection": selection,
        "profiles_table": cluster_profiles,
        "avatar_profiles": out_profiles,
        "report": report,
    }


def save_segmentation(result: dict[str, Any], paths) -> None:
    paths.ensure()
    joblib.dump(result["preprocessor"], paths.seg_preprocessor)
    joblib.dump({"kmeans": result["kmeans"], "features": result["report"]["features"]}, paths.kmeans_model)


def assign_segments(preprocessor: Pipeline, kmeans: KMeans, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Assign new avatars to the fitted segments (validates input columns)."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing segmentation features: {missing}")
    return kmeans.predict(preprocessor.transform(df[features]))
