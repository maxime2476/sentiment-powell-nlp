# -*- coding: utf-8 -*-
"""
03_ngrammes.py
==============
Analyse des collocations : bigrammes et trigrammes.

Fonctions :
  - extraire_bigrammes     : top bigrammes sur le corpus global (PMI / fréquence)
  - extraire_trigrammes    : top trigrammes sur le corpus global
  - ngrammes_vers_dataframe: formate les résultats en DataFrame
  - nuages_bigrammes_par_annee : nuages de bigrammes par année
  - plot_top_ngrammes      : barplot des top-N n-grammes
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from nltk.collocations import (
    BigramAssocMeasures, BigramCollocationFinder,
    TrigramAssocMeasures, TrigramCollocationFinder,
)
from wordcloud import WordCloud

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})


# ---------------------------------------------------------------------------
# Extraction des n-grammes
# ---------------------------------------------------------------------------

def _filtre_ngramme(w: str) -> bool:
    """Retourne True si le mot doit être filtré (trop court ou stopword)."""
    mots_vides_locaux = config.CUSTOM_STOPWORDS
    return len(w) < config.MIN_WORD_LEN or w in mots_vides_locaux


def extraire_bigrammes(
    lemmes: list,
    freq_min: int = config.MIN_FREQ_BIGRAM,
    window: int = config.WINDOW_SIZE,
    top_n: int = config.TOP_N,
) -> list[tuple[tuple, int]]:
    """
    Extrait les top-N bigrammes les plus fréquents du corpus.

    Returns
    -------
    list of ((mot1, mot2), fréquence)
    """
    finder = BigramCollocationFinder.from_words(lemmes, window_size=window)
    finder.apply_word_filter(_filtre_ngramme)
    finder.apply_freq_filter(freq_min)

    candidats = [
        (ng, freq)
        for ng, freq in finder.ngram_fd.items()
        if ng[0] != ng[1]           # pas de répétition du même mot
    ]
    return sorted(candidats, key=lambda x: x[1], reverse=True)[:top_n]


def extraire_trigrammes(
    lemmes: list,
    freq_min: int = config.MIN_FREQ_TRIGRAM,
    window: int = config.WINDOW_SIZE,
    top_n: int = config.TOP_N,
) -> list[tuple[tuple, int]]:
    """
    Extrait les top-N trigrammes les plus fréquents du corpus.

    Returns
    -------
    list of ((mot1, mot2, mot3), fréquence)
    """
    finder = TrigramCollocationFinder.from_words(lemmes, window_size=window)
    finder.apply_word_filter(_filtre_ngramme)
    finder.apply_freq_filter(freq_min)

    candidats = [
        (ng, freq)
        for ng, freq in finder.ngram_fd.items()
        if len(set(ng)) == len(ng)  # tous les mots différents
    ]
    return sorted(candidats, key=lambda x: x[1], reverse=True)[:top_n]


def ngrammes_vers_dataframe(ngrammes: list, n: int) -> pd.DataFrame:
    """Convertit la liste de n-grammes en DataFrame propre."""
    cols = [f"mot_{i+1}" for i in range(n)] + ["frequence"]
    lignes = [list(ng) + [freq] for ng, freq in ngrammes]
    return pd.DataFrame(lignes, columns=cols)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_top_ngrammes(
    df_ng: pd.DataFrame,
    titre: str,
    top_n: int = 20,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Barplot horizontal des top-N n-grammes."""
    # Construction du libellé du n-gramme
    cols_mots = [c for c in df_ng.columns if c.startswith("mot_")]
    df_ng = df_ng.copy()
    df_ng["ngram"] = df_ng[cols_mots].apply(lambda r: " ".join(r), axis=1)
    sous = df_ng.head(top_n).sort_values("frequence", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(sous["ngram"], sous["frequence"], color="steelblue")
    ax.set_title(titre, fontsize=13)
    ax.set_xlabel("Fréquence absolue")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def nuages_bigrammes_par_annee(
    df: pd.DataFrame,
    figures_dir: str = config.FIGURES_DIR,
    show: bool = False,
) -> None:
    """Génère et sauvegarde un nuage de bigrammes par année."""
    mots_vides = config.CUSTOM_STOPWORDS
    annees = sorted(df["year"].unique())

    for annee in annees:
        lemmes_annee = [
            w
            for sous_liste in df.loc[df["year"] == annee, "lemmas"]
            for w in sous_liste
        ]
        if len(lemmes_annee) < 2:
            continue

        finder = BigramCollocationFinder.from_words(
            lemmes_annee, window_size=config.WINDOW_SIZE
        )
        finder.apply_word_filter(_filtre_ngramme)

        comptes = {
            (w1, w2): c
            for (w1, w2), c in finder.ngram_fd.items()
            if w1 != w2
        }
        total = sum(comptes.values())
        if total == 0:
            continue

        freq_rel = {
            f"{w1} {w2}": c / total
            for (w1, w2), c in comptes.items()
        }

        nuage = WordCloud(
            width=1200, height=700,
            background_color="white",
            colormap="Greens",
            max_words=80,
        ).generate_from_frequencies(freq_rel)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(nuage, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Nuage de bigrammes — {annee}", fontsize=16)
        fig.tight_layout()

        save_path = os.path.join(figures_dir, f"wordcloud_bigrammes_{annee}.png")
        fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    print(f"Nuages de bigrammes sauvegardés dans : {figures_dir}")


# ---------------------------------------------------------------------------
# Exécution standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import importlib, sys as _sys
    _sys.path.insert(0, _ROOT)
    from scripts.preprocessing import charger_corpus

    _, lemmes = charger_corpus()

    print("=== Top-20 bigrammes ===")
    bigrams  = extraire_bigrammes(lemmes)
    df_bi    = ngrammes_vers_dataframe(bigrams, n=2)
    print(df_bi.head(20).to_string(index=False))
    plot_top_ngrammes(
        df_bi, "Top-20 bigrammes — Corpus FOMC",
        save_path=os.path.join(config.FIGURES_DIR, "top_bigrammes.png"),
    )

    print("\n=== Top-20 trigrammes ===")
    trigrams = extraire_trigrammes(lemmes)
    df_tri   = ngrammes_vers_dataframe(trigrams, n=3)
    print(df_tri.head(20).to_string(index=False))
    plot_top_ngrammes(
        df_tri, "Top-20 trigrammes — Corpus FOMC",
        save_path=os.path.join(config.FIGURES_DIR, "top_trigrammes.png"),
    )
