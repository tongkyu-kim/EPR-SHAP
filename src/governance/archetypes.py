"""
Governance archetypes: cluster firms on organisational characteristics,
then profile each cluster's responsiveness to each governance mechanism.

Methodology
-----------
- StandardScaler + K-means (default k=4, elbow plotted to assist selection)
- PCA 2D projection for visualisation
- Per-cluster mean feature profile + per-cluster mean CATE per arm
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple

ARCHETYPE_FEATURES = [
    "EXPORT_PCT", "EMP_SIZE", "ESG_REP", "ISO14001",
    "ENV_INSP", "AWR_MEAN", "SUPPLY_POS", "CC_MEAN", "EO_MEAN",
]

CATE_MAP = {
    "Enforcement": "CATE_REGULATORY",
    "Reputation":  "CATE_REPUTATIONAL",
    "Market":      "CATE_MARKET",
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_archetypes(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, List[str]]:
    """
    K-means clustering on standardised firm characteristics.

    Returns
    -------
    df_out    : original df + columns  archetype (int), archetype_label (str)
    km        : fitted KMeans
    scaler    : fitted StandardScaler
    feat_used : list of feature columns actually used
    """
    features = features or ARCHETYPE_FEATURES
    feat_used = [f for f in features if f in df.columns]

    X_raw    = df[feat_used].fillna(df[feat_used].median()).values.astype(float)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = km.fit_predict(X_scaled)

    df_out = df.copy()
    df_out["archetype"]       = labels
    df_out["archetype_label"] = [f"Archetype {i + 1}" for i in labels]

    return df_out, km, scaler, feat_used


def profile_archetypes(
    df_clustered: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Mean feature values and cluster size per archetype.
    """
    features  = features or ARCHETYPE_FEATURES
    feat_used = [f for f in features if f in df_clustered.columns]

    profile = df_clustered.groupby("archetype_label")[feat_used].mean().round(3)
    profile.insert(0, "n_firms", df_clustered.groupby("archetype_label").size())
    return profile


def archetype_treatment_effects(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Mean CATE and optimal mechanism per archetype cluster.
    """
    avail = {k: v for k, v in CATE_MAP.items() if v in df_clustered.columns}
    if not avail:
        return pd.DataFrame()

    te = df_clustered.groupby("archetype_label")[list(avail.values())].mean().round(4)
    te.columns = list(avail.keys())
    te.insert(0, "n_firms", df_clustered.groupby("archetype_label").size())
    te["best_mechanism"] = te[list(avail.keys())].idxmax(axis=1)
    return te


def elbow_scores(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    k_range: range = range(2, 9),
    random_state: int = 42,
) -> pd.DataFrame:
    """Return inertia for each k to support elbow-method cluster selection."""
    features  = features or ARCHETYPE_FEATURES
    feat_used = [f for f in features if f in df.columns]
    X_scaled  = StandardScaler().fit_transform(
        df[feat_used].fillna(df[feat_used].median()).values.astype(float)
    )
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        km.fit(X_scaled)
        rows.append({"k": k, "inertia": km.inertia_})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Auto-naming
# ---------------------------------------------------------------------------

def name_archetypes(profile: pd.DataFrame, te: pd.DataFrame) -> dict:
    """
    Generate short descriptive names from dominant characteristics.
    Returns  {archetype_label: name_string}
    """
    names = {}
    for arch in profile.index:
        row = profile.loc[arch]
        best = te.loc[arch, "best_mechanism"] if arch in te.index else "?"

        export = "Export-oriented"  if row.get("EXPORT_PCT", 0) > 25  else "Domestic"
        size   = "Large firm"       if row.get("EMP_SIZE",   3) >= 4  else "SME"
        esg    = "ESG-active"       if row.get("ESG_REP",    0) > 0.5 else "Non-ESG"

        names[arch] = f"{size}, {export}, {esg} -> {best}"
    return names


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_archetype_profiles(profile: pd.DataFrame, figsize=(13, 6)):
    """
    Parallel-coordinates chart of normalised feature values per archetype.
    """
    import matplotlib.pyplot as plt

    features = [c for c in profile.columns if c != "n_firms"]

    # normalise each feature to [0, 1] across clusters for comparability
    norm = profile[features].copy()
    for col in features:
        lo, hi = norm[col].min(), norm[col].max()
        if hi > lo:
            norm[col] = (norm[col] - lo) / (hi - lo)

    colors = plt.cm.Set2(np.linspace(0, 1, len(norm)))
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(features))

    for i, (arch, row) in enumerate(norm.iterrows()):
        n = int(profile.loc[arch, "n_firms"])
        ax.plot(x, row.values, marker="o", linewidth=2,
                label=f"{arch} (n={n})", color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Normalised feature value (0-1)", fontsize=9)
    ax.set_title(
        "Governance Archetype Feature Profiles\n"
        "(parallel coordinates; normalised within each feature)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_archetype_cates(te: pd.DataFrame, figsize=(9, 5)):
    """Grouped bar chart of mean CATE per archetype x treatment arm."""
    import matplotlib.pyplot as plt

    arm_colors = {"Enforcement": "#e74c3c", "Reputation": "#3498db", "Market": "#2ecc71"}
    arms  = [a for a in arm_colors if a in te.columns]
    archs = te.index.tolist()
    x = np.arange(len(archs))
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    for i, arm in enumerate(arms):
        offset = (i - (len(arms) - 1) / 2) * bar_w
        ax.bar(x + offset, te[arm].values, bar_w,
               label=arm, color=arm_colors[arm], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Mean CATE (BAE_EPR, %)", fontsize=10)
    ax.set_title("Treatment Effects by Governance Archetype", fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(0, linestyle="--", color="gray", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_archetype_pca(
    df_clustered: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize=(8, 6),
):
    """2-D PCA scatter coloured by archetype membership."""
    import matplotlib.pyplot as plt

    features  = features or ARCHETYPE_FEATURES
    feat_used = [f for f in features if f in df_clustered.columns]

    X_scaled = StandardScaler().fit_transform(
        df_clustered[feat_used].fillna(df_clustered[feat_used].median()).values.astype(float)
    )
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    archetypes = sorted(df_clustered["archetype"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(archetypes)))

    fig, ax = plt.subplots(figsize=figsize)
    for arch_id, color in zip(archetypes, colors):
        mask = df_clustered["archetype"] == arch_id
        lab  = df_clustered.loc[mask, "archetype_label"].iloc[0]
        n    = mask.sum()
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], alpha=0.5, s=22, label=f"{lab} (n={n})")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
    ax.set_title("Governance Archetypes in PCA Feature Space", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_elbow(elbow_df: pd.DataFrame, figsize=(6, 4)):
    """Elbow plot to guide cluster-count selection."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o", color="#3498db")
    ax.set_xlabel("Number of clusters (k)", fontsize=10)
    ax.set_ylabel("Inertia (within-cluster SSE)", fontsize=10)
    ax.set_title("K-means Elbow Plot", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
