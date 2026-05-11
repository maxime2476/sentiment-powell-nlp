# -*- coding: utf-8 -*-
"""
01_preprocessing.py
===================
Chargement et pré-traitement du corpus de conférences de presse FOMC.

Pipeline :
  1. Lecture des fichiers .txt du corpus
  2. Extraction de la date depuis le nom de fichier (format YYYYMMDD)
  3. Tokenisation (alphabétique, minuscules)
  4. Étiquetage POS (Part-of-Speech)
  5. Lemmatisation des noms et adjectifs uniquement
  6. Filtrage stopwords (NLTK + personnalisés)
  7. Construction du DataFrame principal et de la liste globale de lemmes

Sortie :
  df_discours   — DataFrame (textes, date, year, tokens, lemmas, texte_nettoye)
  lemmes_filtres — liste plate de tous les lemmes (pour analyses globales)
"""

import os
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Téléchargement des ressources NLTK (silencieux si déjà présentes)
# ---------------------------------------------------------------------------
for _ressource in ("punkt", "punkt_tab", "stopwords",
                   "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
                   "wordnet", "omw-1.4"):
    nltk.download(_ressource, quiet=True)

# ---------------------------------------------------------------------------
# Import de la configuration (support run standalone et import module)
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


def _pos_to_wordnet(tag: str) -> str:
    """Convertit un tag POS Penn Treebank en constante WordNet."""
    if tag.startswith("J"):
        return wordnet.ADJ
    return wordnet.NOUN   # par défaut : nom


def charger_corpus(corpus_dir: str = config.CORPUS_DIR) -> tuple[pd.DataFrame, list]:
    """
    Charge et pré-traite tous les fichiers .txt du dossier corpus.

    Parameters
    ----------
    corpus_dir : str
        Chemin vers le dossier contenant les fichiers .txt du corpus.

    Returns
    -------
    df_discours : pd.DataFrame
        Colonnes : textes, date, year, tokens, lemmas, texte_nettoye
    lemmes_filtres : list[str]
        Liste plate de tous les lemmes du corpus (pour analyses globales).
    """
    lemmatiseur = WordNetLemmatizer()
    mots_vides  = set(stopwords.words("english")) | config.CUSTOM_STOPWORDS

    fichiers = sorted(
        f for f in os.listdir(corpus_dir) if f.endswith(".txt")
    )
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier .txt trouvé dans : {corpus_dir}"
        )

    enregistrements = []

    for nom_fichier in fichiers:
        chemin = os.path.join(corpus_dir, nom_fichier)

        with open(chemin, encoding="utf-8") as fh:
            texte = fh.read()

        # Extraction de la date depuis le nom du fichier
        motif = re.search(r"(\d{8})", nom_fichier)
        if not motif:
            continue
        date  = pd.to_datetime(motif.group(1), format="%Y%m%d")
        annee = date.year

        # Tokenisation — garde uniquement les tokens alphabétiques, en minuscules
        tokens = [
            w for w in word_tokenize(texte.lower(), language="english")
            if w.isalpha()
        ]

        # Étiquetage POS
        etiquettes_pos = nltk.pos_tag(tokens)

        # Lemmatisation sur noms (N*) et adjectifs (J*) de longueur ≥ MIN_WORD_LEN
        lemmes = [
            lemmatiseur.lemmatize(w, pos=_pos_to_wordnet(p))
            for w, p in etiquettes_pos
            if (p.startswith("N") or p.startswith("J"))
            and len(w) >= config.MIN_WORD_LEN
        ]

        # Filtrage des stopwords
        lemmes = [w for w in lemmes if w not in mots_vides]

        enregistrements.append({
            "textes":        nom_fichier,
            "date":          date,
            "year":          annee,
            "tokens":        len(tokens),
            "lemmas":        lemmes,
            "texte_nettoye": " ".join(lemmes),
        })

    df_discours = (
        pd.DataFrame(enregistrements)
        .dropna(subset=["texte_nettoye"])
        .reset_index(drop=True)
    )

    # Liste plate de tous les lemmes (ordre de corpus)
    lemmes_filtres = [
        lemma
        for sous_liste in df_discours["lemmas"]
        for lemma in sous_liste
    ]

    return df_discours, lemmes_filtres


# ---------------------------------------------------------------------------
# Exécution standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Chargement du corpus…")
    df, lemmes = charger_corpus()
    print(f"\n{len(df)} discours chargés ({len(lemmes):,} lemmes au total)\n")
    print(df[["textes", "date", "year", "tokens"]].to_string(index=False))
    print("\nStatistiques des tokens :")
    print(df["tokens"].describe().round(1))
