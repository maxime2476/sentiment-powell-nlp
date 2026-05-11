# -*- coding: utf-8 -*-
"""
04_tfidf.py
===========
Analyse TF-IDF par année (bigrammes et trigrammes).

Le TF-IDF (Term Frequency–Inverse Document Frequency) permet d'identifier
les n-grammes caractéristiques d'une période par rapport au reste du corpus.
Ici, chaque « document » est la concaténation de tous les discours d'une année.

Fonctions :
  - textes_par_annee      : agrégation des lemmes par année
  - calcul_tfidf          : matrice TF-IDF pour une plage de n-grammes
  - top_termes_par_periode: top-N termes TF-IDF par période
  - plot_top_tfidf        : barplot des top-10 termes par période
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def textes_par_annee(df: pd.DataFrame) -> dict:
    """
    Agrège les lemmes par année en un seul texte (document-période).

    Returns
    -------
    dict : {annee: texte_concatene}
    """
    groupes = df.groupby("year")["lemmas"].sum()
    return {
        annee: " ".join(tokens)
        for annee, tokens in groupes.items()
        if tokens
    }


def calcul_tfidf(
    textes: dict,
    ngram_range: tuple = (2, 2),
    max_features: int = config.TFIDF_MAX_FEATURES,
) -> tuple[pd.DataFrame, list]:
    """
    Calcule la matrice TF-IDF pour les textes agrégés par période.

    Returns
    -------
    df_tfidf   : pd.DataFrame  (index=périodes, colonnes=n-grammes)
    periodes   : list          (ordre des périodes)
    """
    periodes = list(textes.keys())
    corps    = list(textes.values())

    vectoriseur = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=1,
        max_df=1.0,
        max_features=max_features,
        stop_words=None,
    )
    X      = vectoriseur.fit_transform(corps)
    traits = vectoriseur.get_feature_names_out()

    # Filtrer les n-grammes contenant des répétitions de mots
    indices_valides = [
        i for i, f in enumerate(traits)
        if len(set(f.split())) == len(f.split())
    ]
    X      = X[:, indices_valides]
    traits = [traits[i] for i in indices_valides]

    df_tfidf = pd.DataFrame(X.toarray(), index=periodes, columns=traits)
    return df_tfidf, periodes


def top_termes_par_periode(
    df_tfidf: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extrait les top-N termes TF-IDF pour chaque période.

    Returns
    -------
    pd.DataFrame (period, term, tfidf)
    """
    lignes = []
    for periode in df_tfidf.index:
        sous = df_tfidf.loc[periode].sort_values(ascending=False).head(top_n)
        for terme, score in sous.items():
            lignes.append({"period": periode, "term": terme, "tfidf": score})
    return pd.DataFrame(lignes)


def plot_top_tfidf(
    df_termes: pd.DataFrame,
    ngram_range: tuple,
    top_n: int = 10,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Barplot des top-N termes TF-IDF par période."""
    sous = df_termes.groupby("period").head(top_n)
    periodes = sorted(sous["period"].unique())
    n_cols = 3
    n_rows = -(-len(periodes) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, p in enumerate(periodes):
        data = sous[sous["period"] == p].sort_values("tfidf", ascending=True)
        axes[i].barh(data["term"], data["tfidf"], color="coral")
        axes[i].set_title(str(p), fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Score TF-IDF")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    label = f"{ngram_range[0]}-gramme{'s' if ngram_range[0] > 1 else ''}"
    fig.suptitle(
        f"Top TF-IDF par année — {label} ({ngram_range[0]}→{ngram_range[1]})",
        fontsize=14, y=1.01,
    )
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
    textes = textes_par_annee(df)

    for ngram, nom in [((2, 2), "bigrammes"), ((3, 3), "trigrammes")]:
        df_tfidf, periodes = calcul_tfidf(textes, ngram_range=ngram)
        df_top = top_termes_par_periode(df_tfidf, top_n=20)

        print(f"\n=== TF-IDF {nom} (top 5 par période) ===")
        print(df_top.groupby("period").head(5).to_string(index=False))

        # Sauvegarde CSV
        csv_path = os.path.join(config.TABLES_DIR, f"tfidf_{nom}.csv")
        df_top.to_csv(csv_path, index=False)

        plot_top_tfidf(
            df_top, ngram_range=ngram,
            save_path=os.path.join(config.FIGURES_DIR, f"tfidf_top_{nom}.png"),
        )
        print(f"Figure sauvegardée : tfidf_top_{nom}.png")
