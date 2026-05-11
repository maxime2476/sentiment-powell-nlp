# -*- coding: utf-8 -*-
"""
08_marches.py
=============
Corrélations entre les discours FOMC et les marchés financiers.

Les discours sont mis en relation avec les performances de :
  - SPY    (ETF S&P 500)
  - NASDAQ (indice composite)
  - BTC    (Bitcoin / USD)

Pour chaque actif et chaque thème macro, on calcule la corrélation de Pearson
entre le score thématique du discours et la variation de prix à différents
horizons temporels : J, J+1, J+3, J+7, J+30.

Fonctions :
  - telecharger_marches          : télécharge les historiques via yfinance
  - enrichir_discours_scores     : ajoute les scores thématiques au DataFrame
  - calculer_correlations        : matrice de corrélations thème × horizon
  - calculer_reaction_par_cluster: variation moyenne de marché par cluster
  - plot_heatmap_correlations    : heatmap corrélations pour un actif
  - plot_reaction_clusters       : barplot réaction marché par cluster
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Téléchargement des données de marché
# ---------------------------------------------------------------------------

def telecharger_marches(
    tickers: dict = config.TICKERS,
    period: str   = config.MARKET_PERIOD,
) -> dict:
    """
    Télécharge les prix de clôture ajustés via yfinance.

    Returns
    -------
    dict : {nom_actif: pd.DataFrame avec colonnes pct_change, pct_change+N}
    """
    dfs_marches = {}

    for nom, ticker in tickers.items():
        try:
            df_prix = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        except Exception as e:
            print(f"[WARN] Impossible de télécharger {ticker} : {e}")
            continue

        if df_prix.empty:
            print(f"[WARN] Données vides pour {ticker}")
            continue

        df_prix.index = pd.to_datetime(df_prix.index).tz_localize(None)

        df_prix["pct_change"]    = df_prix["Close"].pct_change() * 100
        df_prix["pct_change+1"]  = df_prix["Close"].pct_change(periods=1).shift(-1) * 100
        df_prix["pct_change+3"]  = df_prix["Close"].pct_change(periods=3).shift(-3) * 100
        df_prix["pct_change+7"]  = df_prix["Close"].pct_change(periods=7).shift(-7) * 100
        df_prix["pct_change+30"] = df_prix["Close"].pct_change(periods=30).shift(-30) * 100

        dfs_marches[nom] = df_prix
        print(f"  {nom} ({ticker}) : {len(df_prix)} jours téléchargés")

    return dfs_marches


# ---------------------------------------------------------------------------
# Enrichissement du DataFrame discours
# ---------------------------------------------------------------------------

def enrichir_discours_scores(
    df: pd.DataFrame,
    themes: dict = config.THEMES,
) -> pd.DataFrame:
    """
    Ajoute une colonne de score thématique par discours pour chaque thème.

    Returns
    -------
    pd.DataFrame enrichi avec colonnes '{theme}_score'
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for theme, mots_cles in themes.items():
        df[f"{theme}_score"] = df["texte_nettoye"].apply(
            lambda txt: (
                sum(txt.count(k) for k in mots_cles) / max(len(txt.split()), 1)
            )
        )
    return df


# ---------------------------------------------------------------------------
# Calcul des corrélations
# ---------------------------------------------------------------------------

def calculer_correlations(
    df_discours: pd.DataFrame,
    dfs_marches: dict,
    themes: dict     = config.THEMES,
    horizons: list   = config.HORIZONS,
) -> dict:
    """
    Calcule les corrélations thème × horizon pour chaque actif.

    Returns
    -------
    dict : {nom_actif: pd.DataFrame (horizons en index, thèmes en colonnes)}
    """
    corrs_tous = {}

    for nom_actif, df_marche in dfs_marches.items():
        df_fusion = pd.merge(
            df_discours, df_marche[horizons],
            left_on="date", right_index=True, how="left",
        )

        corrs_themes = {}
        for theme in themes.keys():
            col = f"{theme}_score"
            if col in df_fusion.columns and not df_fusion[col].isna().all():
                corrs_themes[theme] = (
                    df_fusion[[col] + horizons]
                    .corr()[col][horizons]
                )

        if corrs_themes:
            corrs_tous[nom_actif] = pd.DataFrame(corrs_themes)

    return corrs_tous


# ---------------------------------------------------------------------------
# Réaction des marchés par cluster
# ---------------------------------------------------------------------------

def calculer_reaction_par_cluster(
    df_discours: pd.DataFrame,
    df_marche: pd.DataFrame,
    horizons: list = config.HORIZONS,
) -> pd.DataFrame:
    """
    Calcule la variation de marché moyenne par cluster pour un actif donné.

    Returns
    -------
    pd.DataFrame (index=cluster, colonnes=horizons)
    """
    df_fusion = pd.merge(
        df_discours[["date", "cluster"]],
        df_marche[horizons],
        left_on="date", right_index=True, how="left",
    )

    return (
        df_fusion.groupby("cluster")[horizons]
        .mean()
        .sort_index()
        .round(4)
    )


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_heatmap_correlations(
    df_corr: pd.DataFrame,
    nom_actif: str,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Heatmap des corrélations thème × horizon pour un actif."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        df_corr.T,
        annot=True, fmt=".3f",
        cmap="coolwarm", center=0,
        linewidths=0.5, ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title(
        f"Corrélations thèmes FOMC × variations {nom_actif}\n"
        f"(horizons : J, J+1, J+3, J+7, J+30)",
        fontsize=12,
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Thème")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_reaction_clusters(
    df_reaction: pd.DataFrame,
    nom_actif: str,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Heatmap de la réaction marché moyenne par cluster et horizon."""
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        df_reaction,
        annot=True, fmt=".3f",
        cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(
        f"Variation moyenne de {nom_actif} par cluster de discours",
        fontsize=12,
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Cluster")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_correlations_par_actif(
    corrs_tous: dict,
    save_dir: str = config.FIGURES_DIR,
    show: bool = True,
) -> None:
    """Génère et sauvegarde une heatmap de corrélations pour chaque actif."""
    for nom_actif, df_corr in corrs_tous.items():
        save_path = os.path.join(save_dir, f"correlations_{nom_actif}.png")
        plot_heatmap_correlations(df_corr, nom_actif, save_path=save_path, show=show)
        print(f"Heatmap sauvegardée : correlations_{nom_actif}.png")


# ---------------------------------------------------------------------------
# Exécution standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scripts.preprocessing import charger_corpus
    from scripts.afc_clustering import projection_svd, clustering_kmeans

    df, _ = charger_corpus()

    print("Téléchargement des données de marché…")
    dfs_marches = telecharger_marches()

    df_enrichi = enrichir_discours_scores(df)

    print("\n=== Calcul des corrélations ===")
    corrs = calculer_correlations(df_enrichi, dfs_marches)
    for actif, df_corr in corrs.items():
        print(f"\n--- {actif} ---")
        print(df_corr.round(4))

    plot_correlations_par_actif(corrs)

    # Réaction par cluster
    df_svd, vec, svd_m = projection_svd(df)
    df_svd = clustering_kmeans(df_svd)
    df_enrichi["cluster"] = df_svd["cluster"].values

    for nom_actif, df_marche in dfs_marches.items():
        df_reaction = calculer_reaction_par_cluster(df_enrichi, df_marche)
        print(f"\n=== Réaction {nom_actif} par cluster ===")
        print(df_reaction)
        plot_reaction_clusters(
            df_reaction, nom_actif,
            save_path=os.path.join(
                config.FIGURES_DIR, f"reaction_clusters_{nom_actif}.png"
            ),
        )
