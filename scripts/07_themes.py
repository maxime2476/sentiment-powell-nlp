# -*- coding: utf-8 -*-
"""
07_themes.py
============
Analyse de l'évolution thématique macro-économique.

Quatre thèmes sont suivis dans les discours FOMC :
  - inflation : prix, coûts, demande/offre
  - labor     : emploi, chômage, marché du travail
  - growth    : croissance, PIB, investissement
  - rates     : taux d'intérêt, politique monétaire, crédit

La fréquence relative de chaque thème est calculée sur les textes
agrégés par année, permettant d'observer les priorités de la Fed.

Fonctions :
  - calculer_scores_themes    : fréquences relatives par thème et par année
  - plot_evolution_themes     : lineplot d'évolution
  - plot_heatmap_themes       : heatmap thème × année
  - plot_radar_themes         : graphique radar par année
  - plot_barplot_themes       : barplot empilé de composition thématique
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

PALETTE_THEMES = {
    "inflation": "#e74c3c",
    "labor":     "#f39c12",
    "growth":    "#2ecc71",
    "rates":     "#3498db",
}


# ---------------------------------------------------------------------------
# Fonctions d'analyse
# ---------------------------------------------------------------------------

def calculer_scores_themes(
    df: pd.DataFrame,
    themes: dict = config.THEMES,
) -> pd.DataFrame:
    """
    Calcule la fréquence relative de chaque thème macro par année.

    Parameters
    ----------
    df     : DataFrame avec colonnes 'year' et 'lemmas'
    themes : dict {theme: [mots_cles]}

    Returns
    -------
    pd.DataFrame (year, theme, freq)
    """
    lemmes_par_annee = df.groupby("year")["lemmas"].sum()

    lignes = []
    for annee, lemmes in lemmes_par_annee.items():
        texte = " ".join(lemmes)
        total = len(texte.split())
        if total == 0:
            continue
        for theme, mots_cles in themes.items():
            freq = sum(texte.count(k) for k in mots_cles) / total
            lignes.append({"year": annee, "theme": theme, "freq": freq})

    return pd.DataFrame(lignes)


def tableau_pivot_themes(df_themes: pd.DataFrame) -> pd.DataFrame:
    """Retourne un tableau croisé année × thème."""
    return (
        df_themes.pivot(index="year", columns="theme", values="freq")
        .round(5)
    )


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_evolution_themes(
    df_themes: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Lineplot de l'évolution thématique par année."""
    fig, ax = plt.subplots(figsize=(13, 6))

    for theme in df_themes["theme"].unique():
        sous = df_themes[df_themes["theme"] == theme]
        couleur = PALETTE_THEMES.get(theme, None)
        ax.plot(sous["year"], sous["freq"], marker="o",
                label=theme.capitalize(), color=couleur, linewidth=2)

    ax.set_title(
        "Évolution thématique : inflation, travail, croissance, taux\n"
        "(fréquence relative — conférences FOMC 2020–2025)",
        fontsize=13,
    )
    ax.set_xlabel("Année")
    ax.set_ylabel("Fréquence relative")
    ax.legend(title="Thème", fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_heatmap_themes(
    df_themes: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Heatmap thème × année."""
    pivot = tableau_pivot_themes(df_themes)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        pivot.T,
        annot=True, fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Heatmap thématique — Conférences FOMC", fontsize=13)
    ax.set_xlabel("Année")
    ax.set_ylabel("Thème")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_barplot_themes(
    df_themes: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Barplot empilé normalisé : composition thématique par année."""
    pivot = tableau_pivot_themes(df_themes)
    # Normalisation : part de chaque thème dans le total de l'année
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100
    couleurs   = [PALETTE_THEMES.get(c, "#aaa") for c in pivot_norm.columns]

    fig, ax = plt.subplots(figsize=(13, 6))
    pivot_norm.plot(kind="bar", stacked=True, ax=ax, color=couleurs,
                    edgecolor="white", width=0.7)
    ax.set_title(
        "Composition thématique par année (part relative) — FOMC",
        fontsize=13,
    )
    ax.set_xlabel("Année")
    ax.set_ylabel("Part (%) du total thématique")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Thème", loc="upper left", fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_radar_themes(
    df_themes: pd.DataFrame,
    annees_selectionnees: list | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Graphique radar (spider) pour comparer les thèmes entre années."""
    pivot   = tableau_pivot_themes(df_themes)
    annees  = annees_selectionnees or list(pivot.index)
    themes  = list(pivot.columns)
    n       = len(themes)
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # fermeture du polygone

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    cmap = plt.get_cmap("tab10")
    for i, annee in enumerate(annees):
        valeurs = pivot.loc[annee, themes].tolist()
        valeurs += valeurs[:1]
        ax.plot(angles, valeurs, linewidth=2, label=str(annee), color=cmap(i))
        ax.fill(angles, valeurs, alpha=0.1, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.capitalize() for t in themes], fontsize=11)
    ax.set_title("Radar thématique — FOMC", fontsize=13, pad=20)
    ax.legend(title="Année", loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

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
    df_themes = calculer_scores_themes(df)

    print("=== Pivot thèmes × années ===")
    print(tableau_pivot_themes(df_themes))

    df_themes.to_csv(os.path.join(config.TABLES_DIR, "scores_themes.csv"), index=False)

    plot_evolution_themes(
        df_themes,
        save_path=os.path.join(config.FIGURES_DIR, "evolution_themes.png"),
    )
    plot_heatmap_themes(
        df_themes,
        save_path=os.path.join(config.FIGURES_DIR, "heatmap_themes.png"),
    )
    plot_barplot_themes(
        df_themes,
        save_path=os.path.join(config.FIGURES_DIR, "barplot_themes.png"),
    )
    plot_radar_themes(
        df_themes,
        save_path=os.path.join(config.FIGURES_DIR, "radar_themes.png"),
    )
