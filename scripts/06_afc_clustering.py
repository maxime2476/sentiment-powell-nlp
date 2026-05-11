# -*- coding: utf-8 -*-
"""
06_afc_clustering.py
====================
Analyse Factorielle des Correspondances (via SVD tronquée) et clustering.

Pipeline :
  1. Vectorisation TF-IDF des discours individuels
  2. Réduction de dimension (TruncatedSVD, 2 composantes) → projection AFC
  3. Clustering K-Means (3 clusters) sur les coordonnées normalisées
  4. Caractérisation des clusters : scores hawkish / dovish
  5. Analyse de la réaction du marché par cluster (connexion avec 08_marches)

Fonctions :
  - projection_svd          : TF-IDF + SVD → coordonnées 2D
  - clustering_kmeans       : K-Means sur les coordonnées
  - caracteriser_clusters   : scores hawkish/dovish par cluster
  - plot_afc                : scatterplot de la projection AFC
  - plot_clusters           : scatterplot colorié par cluster
  - plot_top_mots_svd       : mots les plus influents sur chaque axe
  - plot_hawkish_dovish     : barplot hawkish vs dovish par cluster
"""

import os
import sys

# Évite le warning KMeans / MKL sur Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

PALETTE_CLUSTERS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


# ---------------------------------------------------------------------------
# Projection SVD (AFC)
# ---------------------------------------------------------------------------

def projection_svd(
    df: pd.DataFrame,
    n_components: int = config.SVD_COMPONENTS,
    max_features: int = config.TFIDF_MAX_FEATURES,
) -> tuple[pd.DataFrame, TfidfVectorizer, TruncatedSVD]:
    """
    Vectorise les discours (TF-IDF) puis projette en 2D via SVD tronquée.

    Returns
    -------
    df_svd      : pd.DataFrame  (Dim1, Dim2, year, textes)
    vectoriseur : TfidfVectorizer ajusté
    svd         : TruncatedSVD ajusté
    """
    vectoriseur = TfidfVectorizer(max_features=max_features)
    X           = vectoriseur.fit_transform(df["texte_nettoye"])

    svd         = TruncatedSVD(n_components=n_components, random_state=42)
    coordonnees = svd.fit_transform(X)

    df_svd = pd.DataFrame(
        coordonnees,
        columns=[f"Dim{i+1}" for i in range(n_components)],
    )
    df_svd["year"]   = df["year"].values
    df_svd["textes"] = df["textes"].values

    variance_exp = svd.explained_variance_ratio_ * 100
    print(
        f"Variance expliquée — Dim1: {variance_exp[0]:.1f}%  "
        f"Dim2: {variance_exp[1]:.1f}%  "
        f"Total: {variance_exp.sum():.1f}%"
    )
    return df_svd, vectoriseur, svd


def top_mots_composantes(
    vectoriseur: TfidfVectorizer,
    svd: TruncatedSVD,
    top_n: int = 10,
) -> dict:
    """
    Retourne les top-N mots qui contribuent le plus à chaque composante SVD.

    Returns
    -------
    dict : {"Dim1_pos": [...], "Dim1_neg": [...], "Dim2_pos": [...], ...}
    """
    termes     = vectoriseur.get_feature_names_out()
    resultats  = {}

    for i, composante in enumerate(svd.components_):
        indices_pos = np.argsort(composante)[-top_n:][::-1]
        indices_neg = np.argsort(composante)[:top_n]
        resultats[f"Dim{i+1}_pos"] = [termes[j] for j in indices_pos]
        resultats[f"Dim{i+1}_neg"] = [termes[j] for j in indices_neg]

    return resultats


# ---------------------------------------------------------------------------
# Clustering K-Means
# ---------------------------------------------------------------------------

def clustering_kmeans(
    df_svd: pd.DataFrame,
    n_clusters: int = config.N_CLUSTERS,
) -> pd.DataFrame:
    """
    Applique K-Means sur les coordonnées SVD normalisées.

    Returns
    -------
    df_svd enrichi d'une colonne 'cluster'
    """
    coords     = df_svd[["Dim1", "Dim2"]].values
    X_norme    = StandardScaler().fit_transform(coords)
    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_svd     = df_svd.copy()
    df_svd["cluster"] = kmeans.fit_predict(X_norme)
    return df_svd


def caracteriser_clusters(
    df: pd.DataFrame,
    df_svd: pd.DataFrame,
    hawkish_words: list = config.HAWKISH_WORDS,
    dovish_words: list  = config.DOVISH_WORDS,
) -> pd.DataFrame:
    """
    Caractérise chaque cluster par ses scores hawkish et dovish (via TF-IDF).

    Returns
    -------
    pd.DataFrame (cluster, hawkish, dovish, orientation)
    """
    df_joint = df.copy()
    df_joint["cluster"] = df_svd["cluster"].values

    textes_cluster = df_joint.groupby("cluster")["texte_nettoye"].apply(
        lambda x: " ".join(x)
    )

    tfidf = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES, stop_words="english")
    X     = tfidf.fit_transform(textes_cluster)
    termes = tfidf.get_feature_names_out()
    df_tfidf_cluster = pd.DataFrame(
        X.toarray(), index=textes_cluster.index, columns=termes
    )

    scores = []
    for cluster, ligne in df_tfidf_cluster.iterrows():
        hawk = ligne[[w for w in ligne.index if w in hawkish_words]].sum()
        dov  = ligne[[w for w in ligne.index if w in dovish_words]].sum()
        orientation = "Hawkish" if hawk > dov else "Dovish"
        scores.append({
            "cluster":     cluster,
            "hawkish":     round(hawk, 6),
            "dovish":      round(dov, 6),
            "orientation": orientation,
        })

    return pd.DataFrame(scores)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_afc(
    df_svd: pd.DataFrame,
    vectoriseur: TfidfVectorizer,
    svd: TruncatedSVD,
    top_mots: int = 5,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Scatterplot AFC colorié par année + labels des mots les plus influents."""
    termes       = vectoriseur.get_feature_names_out()
    composantes  = svd.components_
    importance   = np.sum(np.abs(composantes), axis=0)
    top_indices  = np.argsort(importance)[-top_mots:][::-1]

    # Normalisation des coordonnées des mots dans l'espace discours
    x_min, x_max = df_svd["Dim1"].min(), df_svd["Dim1"].max()
    y_min, y_max = df_svd["Dim2"].min(), df_svd["Dim2"].max()

    comp_x = composantes[0, :]
    comp_y = composantes[1, :]
    cx_norm = (comp_x - comp_x.min()) / (comp_x.max() - comp_x.min()) * (x_max - x_min) + x_min
    cy_norm = (comp_y - comp_y.min()) / (comp_y.max() - comp_y.min()) * (y_max - y_min) + y_min

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=df_svd, x="Dim1", y="Dim2",
        hue="year", palette="viridis", s=80, ax=ax,
    )
    for idx in top_indices:
        ax.text(
            cx_norm[idx], cy_norm[idx], termes[idx],
            fontsize=11, fontweight="bold", color="crimson",
            ha="center", va="center",
        )

    var = svd.explained_variance_ratio_ * 100
    ax.set_title("AFC — Projection sémantique des discours FOMC", fontsize=13)
    ax.set_xlabel(f"Dim 1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"Dim 2 ({var[1]:.1f}%)")
    ax.legend(title="Année", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_clusters(
    df_svd: pd.DataFrame,
    df_scores: pd.DataFrame | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Scatterplot AFC colorié par cluster K-Means."""
    n_clusters = df_svd["cluster"].nunique()
    palette = PALETTE_CLUSTERS[:n_clusters]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df_svd, x="Dim1", y="Dim2",
        hue="cluster", palette=palette,
        s=90, ax=ax,
    )

    # Annotation des centroïdes
    for cl in sorted(df_svd["cluster"].unique()):
        cx = df_svd.loc[df_svd["cluster"] == cl, "Dim1"].mean()
        cy = df_svd.loc[df_svd["cluster"] == cl, "Dim2"].mean()
        label = f"C{cl}"
        if df_scores is not None:
            row = df_scores[df_scores["cluster"] == cl]
            if not row.empty:
                label += f"\n({row['orientation'].values[0]})"
        ax.text(cx, cy, label, fontsize=11, fontweight="bold",
                color="black", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_title(f"Clusters K-Means (k={n_clusters}) — Projection AFC", fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(title="Cluster", fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_hawkish_dovish(
    df_scores: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Barplot comparatif hawkish vs dovish par cluster."""
    df_melt = df_scores.melt(
        id_vars="cluster", value_vars=["hawkish", "dovish"],
        var_name="type", value_name="score",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df_melt, x="cluster", y="score",
        hue="type", palette={"hawkish": "#e74c3c", "dovish": "#3498db"},
        ax=ax,
    )
    ax.set_title("Score Hawkish vs. Dovish par cluster (TF-IDF)", fontsize=13)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Score TF-IDF agrégé")
    ax.legend(title="Orientation")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_distribution_clusters_par_annee(
    df: pd.DataFrame,
    df_svd: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Barplot empilé : répartition des discours par cluster et par année."""
    df_joint = df[["year"]].copy()
    df_joint["cluster"] = df_svd["cluster"].values

    pivot = (
        df_joint.groupby(["year", "cluster"])
        .size()
        .unstack(fill_value=0)
    )
    n_clusters = pivot.columns.nunique()
    couleurs   = PALETTE_CLUSTERS[:n_clusters]

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=couleurs, edgecolor="white")
    ax.set_title("Répartition des discours par cluster et par année", fontsize=13)
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre de discours")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Cluster", fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Exécution standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scripts.preprocessing import charger_corpus

    df, _ = charger_corpus()

    df_svd, vec, svd_model = projection_svd(df)
    df_svd = clustering_kmeans(df_svd)
    df_scores = caracteriser_clusters(df, df_svd)

    print("\n=== Orientation hawkish/dovish par cluster ===")
    print(df_scores.to_string(index=False))

    print("\n=== Composition des clusters par année ===")
    df_comp = df[["year"]].copy()
    df_comp["cluster"] = df_svd["cluster"].values
    print(df_comp.groupby(["year","cluster"]).size().unstack(fill_value=0))

    # Mots influents
    top_mots = top_mots_composantes(vec, svd_model)
    print("\n=== Mots les plus influents ===")
    for dim, mots in top_mots.items():
        print(f"  {dim}: {', '.join(mots)}")

    plot_afc(
        df_svd, vec, svd_model,
        save_path=os.path.join(config.FIGURES_DIR, "afc_projection.png"),
    )
    plot_clusters(
        df_svd, df_scores,
        save_path=os.path.join(config.FIGURES_DIR, "kmeans_clusters.png"),
    )
    plot_hawkish_dovish(
        df_scores,
        save_path=os.path.join(config.FIGURES_DIR, "hawkish_dovish.png"),
    )
    plot_distribution_clusters_par_annee(
        df, df_svd,
        save_path=os.path.join(config.FIGURES_DIR, "distribution_clusters_annee.png"),
    )
