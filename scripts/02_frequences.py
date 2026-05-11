# -*- coding: utf-8 -*-
"""
02_frequences.py
================
Analyse des fréquences lexicales et nuages de mots.

Fonctions :
  - statistiques_tokens        : résumé descriptif des longueurs de discours
  - top_mots_par_annee         : top-N mots (fréquence relative) par année
  - plot_distribution_tokens   : histogramme de la distribution des tokens
  - plot_evolution_tokens      : évolution temporelle du nombre de tokens
  - plot_top10_barplot         : barplot des top-10 mots par année
  - generer_nuages_mots_unigrammes : nuages de mots unigrammes par année
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

# ---------------------------------------------------------------------------
# Style Matplotlib global
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})


# ---------------------------------------------------------------------------
# Fonctions d'analyse
# ---------------------------------------------------------------------------

def statistiques_tokens(df: pd.DataFrame) -> pd.Series:
    """Retourne les statistiques descriptives du nombre de tokens par discours."""
    stats = df["tokens"].describe().round(1)
    return stats


def top_mots_par_annee(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Calcule les top-N mots les plus fréquents (fréquence relative) par année.

    Returns
    -------
    dict : {annee: [(mot, freq_relative), ...]}
    """
    annees = sorted(df["year"].unique())
    resultats = {}

    for annee in annees:
        lemmes_annee = [
            w
            for sous_liste in df.loc[df["year"] == annee, "lemmas"]
            for w in sous_liste
        ]
        total = len(lemmes_annee)
        if total == 0:
            continue
        compteur = Counter(lemmes_annee)
        freq_rel  = {w: c / total for w, c in compteur.items()}
        top       = sorted(freq_rel.items(), key=lambda x: x[1], reverse=True)[:top_n]
        resultats[annee] = top

    return resultats


def top_mots_vers_dataframe(top_mots: dict) -> pd.DataFrame:
    """Convertit le dictionnaire top_mots_par_annee en DataFrame."""
    lignes = []
    for annee, liste in top_mots.items():
        for rang, (mot, freq) in enumerate(liste, start=1):
            lignes.append({"year": annee, "word": mot, "freq": freq, "rank": rang})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_distribution_tokens(
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Histogramme + KDE de la distribution du nombre de tokens par discours."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["tokens"], bins=10, kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribution du nombre de tokens par discours (FOMC)", fontsize=14)
    ax.set_xlabel("Nombre de tokens")
    ax.set_ylabel("Fréquence")

    # annotations : min, max, médiane
    med = df["tokens"].median()
    ax.axvline(med, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Médiane : {med:,.0f}")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_evolution_tokens(
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Évolution temporelle du nombre de tokens par discours (lineplot)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=df.sort_values("date"),
        x="date", y="tokens",
        marker="o", color="steelblue", ax=ax,
    )
    ax.set_title("Évolution du nombre de tokens par conférence de presse (FOMC)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Nombre de tokens")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_top10_barplot(
    df_top: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Barplot des top-10 mots par année (fréquence relative)."""
    annees = sorted(df_top["year"].unique())
    n_cols = 3
    n_rows = -(-len(annees) // n_cols)   # division entière par excès

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, annee in enumerate(annees):
        sous = df_top[df_top["year"] == annee].sort_values("freq", ascending=True)
        axes[i].barh(sous["word"], sous["freq"], color="steelblue")
        axes[i].set_title(str(annee), fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Fréquence relative")

    # Masquer les axes inutilisés
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top-10 mots par année — Conférences FOMC", fontsize=14, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def generer_nuages_mots_unigrammes(
    df: pd.DataFrame,
    figures_dir: str = config.FIGURES_DIR,
    show: bool = False,
) -> None:
    """Génère et sauvegarde un nuage de mots par année."""
    annees = sorted(df["year"].unique())

    for annee in annees:
        lemmes_annee = [
            w
            for sous_liste in df.loc[df["year"] == annee, "lemmas"]
            for w in sous_liste
        ]
        total = len(lemmes_annee)
        if total == 0:
            continue

        compteur  = Counter(lemmes_annee)
        freq_rel  = {w: c / total for w, c in compteur.items()}

        nuage = WordCloud(
            width=1200, height=700,
            background_color="white",
            colormap="Blues",
            max_words=120,
        ).generate_from_frequencies(freq_rel)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(nuage, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Nuage de mots — {annee}", fontsize=16)
        fig.tight_layout()

        save_path = os.path.join(figures_dir, f"wordcloud_unigrammes_{annee}.png")
        fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    print(f"Nuages de mots sauvegardés dans : {figures_dir}")


# ---------------------------------------------------------------------------
# Exécution standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scripts.preprocessing import charger_corpus   # noqa: E402  (évite import circulaire)

    df, _ = charger_corpus()

    print("=== Statistiques tokens ===")
    print(statistiques_tokens(df))

    print("\n=== Top-5 textes (plus longs) ===")
    print(df[["textes", "tokens"]].sort_values("tokens", ascending=False).head())

    print("\n=== Top-5 textes (plus courts) ===")
    print(df[["textes", "tokens"]].sort_values("tokens").head())

    plot_distribution_tokens(
        df,
        save_path=os.path.join(config.FIGURES_DIR, "distribution_tokens.png"),
    )
    plot_evolution_tokens(
        df,
        save_path=os.path.join(config.FIGURES_DIR, "evolution_tokens.png"),
    )

    top_mots = top_mots_par_annee(df, top_n=10)
    df_top   = top_mots_vers_dataframe(top_mots)
    print("\n=== Top mots par année ===")
    print(df_top)

    plot_top10_barplot(
        df_top,
        save_path=os.path.join(config.FIGURES_DIR, "top10_mots_par_annee.png"),
    )
    generer_nuages_mots_unigrammes(df)
