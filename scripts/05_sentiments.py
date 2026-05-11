# -*- coding: utf-8 -*-
"""
05_sentiments.py
================
Analyse des tonalités économiques par lexique.

Quatre tonalités sont étudiées :
  - optimisme   : mots associés à une perspective positive
  - pessimisme  : mots associés à une dégradation économique
  - incertitude : mots associés à l'imprévisibilité et aux risques
  - stabilité   : mots associés à l'équilibre et à la maîtrise

La fréquence relative de chaque tonalité est calculée sur les textes
agrégés par année, puis tracée pour visualiser l'évolution temporelle.

Fonctions :
  - calculer_scores_sentiments : fréquences relatives par tonalité et par année
  - plot_evolution_sentiments  : lineplot de l'évolution des tonalités
  - plot_heatmap_sentiments    : heatmap tonalité × année
  - tableau_pivot_sentiments   : tableau croisé pour export
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

# Palette fixe pour cohérence visuelle inter-graphiques
PALETTE_SENTIMENTS = {
    "optimism":    "#2ecc71",
    "pessimism":   "#e74c3c",
    "uncertainty": "#e67e22",
    "stability":   "#3498db",
}


# ---------------------------------------------------------------------------
# Fonctions d'analyse
# ---------------------------------------------------------------------------

def calculer_scores_sentiments(
    df: pd.DataFrame,
    sentiments: dict = config.SENTIMENTS,
) -> pd.DataFrame:
    """
    Calcule la fréquence relative de chaque tonalité par année.

    Parameters
    ----------
    df         : DataFrame avec colonnes 'year' et 'lemmas'
    sentiments : dict {tonalite: [mots_cles]}

    Returns
    -------
    pd.DataFrame (year, sentiment, freq)
    """
    lemmes_par_annee = df.groupby("year")["lemmas"].sum()

    lignes = []
    for annee, lemmes in lemmes_par_annee.items():
        texte = " ".join(lemmes)
        total = len(texte.split())
        if total == 0:
            continue
        for sentiment, mots_cles in sentiments.items():
            freq = sum(texte.count(k) for k in mots_cles) / total
            lignes.append({"year": annee, "sentiment": sentiment, "freq": freq})

    return pd.DataFrame(lignes)


def tableau_pivot_sentiments(df_sent: pd.DataFrame) -> pd.DataFrame:
    """Retourne un tableau croisé année × tonalité (fréquence relative)."""
    return (
        df_sent.pivot(index="year", columns="sentiment", values="freq")
        .round(5)
    )


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_evolution_sentiments(
    df_sent: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Lineplot de l'évolution des tonalités économiques par année."""
    fig, ax = plt.subplots(figsize=(13, 6))

    for sentiment in df_sent["sentiment"].unique():
        sous = df_sent[df_sent["sentiment"] == sentiment]
        couleur = PALETTE_SENTIMENTS.get(sentiment, None)
        ax.plot(sous["year"], sous["freq"], marker="o",
                label=sentiment.capitalize(), color=couleur, linewidth=2)

    ax.set_title(
        "Évolution des tonalités économiques dans les discours FOMC (2020–2025)",
        fontsize=13,
    )
    ax.set_xlabel("Année")
    ax.set_ylabel("Fréquence relative")
    ax.legend(title="Tonalité", fontsize=10)
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_heatmap_sentiments(
    df_sent: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Heatmap tonalité × année (fréquence relative)."""
    pivot = tableau_pivot_sentiments(df_sent)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        pivot.T,
        annot=True, fmt=".4f",
        cmap="RdYlGn",
        center=pivot.values.mean(),
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Heatmap des tonalités économiques — FOMC", fontsize=13)
    ax.set_xlabel("Année")
    ax.set_ylabel("Tonalité")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_area_sentiments(
    df_sent: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Area stacked plot de la composition des tonalités par année."""
    pivot = tableau_pivot_sentiments(df_sent)
    couleurs = [PALETTE_SENTIMENTS.get(c, "#aaa") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(13, 6))
    pivot.plot.area(ax=ax, color=couleurs, alpha=0.75)
    ax.set_title("Composition des tonalités économiques par année — FOMC", fontsize=13)
    ax.set_xlabel("Année")
    ax.set_ylabel("Fréquence relative cumulée")
    ax.legend(title="Tonalité", loc="upper left", fontsize=10)

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
    df_sent = calculer_scores_sentiments(df)

    print("=== Scores de tonalité (pivot) ===")
    print(tableau_pivot_sentiments(df_sent))

    # Sauvegarde CSV
    df_sent.to_csv(os.path.join(config.TABLES_DIR, "scores_sentiments.csv"), index=False)

    plot_evolution_sentiments(
        df_sent,
        save_path=os.path.join(config.FIGURES_DIR, "evolution_sentiments.png"),
    )
    plot_heatmap_sentiments(
        df_sent,
        save_path=os.path.join(config.FIGURES_DIR, "heatmap_sentiments.png"),
    )
    plot_area_sentiments(
        df_sent,
        save_path=os.path.join(config.FIGURES_DIR, "area_sentiments.png"),
    )
