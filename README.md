# Analyse Textuelle des Conférences de Presse du FOMC (2020–2025)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-00B4CC?style=flat-square&logo=nltk&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.25%2B-013243?style=flat-square&logo=numpy&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-0.2%2B-purple?style=flat-square)
![License](https://img.shields.io/badge/Licence-Académique-lightgrey?style=flat-square)

> **Devoir Maison — M2 Économie Appliquée**  
> Maxime Gourguechon · Analyse quantitative textuelle

---

## Présentation

Ce projet applique des méthodes d'**analyse quantitative textuelle** (NLP) au corpus des **retranscriptions des conférences de presse du Federal Open Market Committee (FOMC)** de la Réserve fédérale américaine sur la période 2020–2025.

L'objectif est de quantifier l'évolution du discours de la Fed, d'en extraire les thèmes et tonalités dominants, et de mesurer l'impact de la rhétorique monétaire sur les marchés financiers (S&P 500, NASDAQ, Bitcoin).

---

## Corpus

| Paramètre | Valeur |
|-----------|--------|
| Source | Transcriptions officielles FOMC (format `.txt`) |
| Période | Septembre 2020 – Juillet 2025 |
| Nombre de documents | 40 conférences de presse |
| Fréquence | ~8 conférences par an (toutes les 6–8 semaines) |
| Langue | Anglais |

---

## Structure du projet

```
dm-analyse-textuelle-fomc/
│
├── config.py                     # Configuration centralisée (chemins, paramètres, lexiques)
├── requirements.txt              # Dépendances Python
├── README.md
├── .gitignore
│
├── archive/                      # Corpus FOMC (.txt) — non versionné (voir .gitignore)
│   ├── FOMCpresconf20200916.txt
│   └── ...
│
├── scripts/                      # Modules Python autonomes
│   ├── __init__.py
│   ├── 01_preprocessing.py       # Chargement, tokenisation, lemmatisation, filtrage
│   ├── 02_frequences.py          # Fréquences lexicales, nuages de mots
│   ├── 03_ngrammes.py            # Bigrammes et trigrammes (collocations)
│   ├── 04_tfidf.py               # TF-IDF par période
│   ├── 05_sentiments.py          # Tonalités économiques (lexique)
│   ├── 06_afc_clustering.py      # SVD/AFC + K-Means
│   ├── 07_themes.py              # Évolution thématique macro
│   └── 08_marches.py             # Corrélations avec les marchés financiers
│
├── notebooks/
│   └── analyse_fomc.ipynb        # Notebook explicatif complet avec graphiques
│
└── outputs/
    ├── figures/                  # Figures générées (.png) — non versionnées
    └── tables/                   # Tableaux exportés (.csv) — non versionnés
```

---

## Pipeline NLP

```
┌─────────────────────────────────────────────────────────────┐
│                      Corpus FOMC (.txt)                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │ 01 Prétraitement                         │
                    │  • Tokenisation (alpha, lowercase)       │
                    │  • POS tagging (Penn Treebank)           │
                    │  • Lemmatisation (noms + adjectifs)      │
                    │  • Filtrage stopwords (NLTK + perso)     │
                    └─────┬─────┘
                          │  df_discours, lemmes_filtres
          ┌───────────────┼──────────────────────┐
          │               │                      │
    ┌─────▼─────┐   ┌─────▼─────┐         ┌─────▼─────┐
    │ 02 Fréq.  │   │ 03 N-gram │         │ 04 TF-IDF │
    │  Wordcloud│   │ Bi / Tri  │         │ par année │
    └───────────┘   └───────────┘         └───────────┘
          │
    ┌─────▼──────────────────────────────────────┐
    │             05 Sentiments                   │
    │  Optimisme / Pessimisme / Incertitude / Stabilité│
    └─────┬──────────────────────────────────────┘
          │
    ┌─────▼──────────────────────────────────────┐
    │     06 AFC (SVD) + K-Means (k=3)            │
    │  Hawkish / Dovish par cluster               │
    └─────┬──────────────────────────────────────┘
          │
    ┌─────▼──────────────────────────────────────┐
    │     07 Thèmes                               │
    │  Inflation / Labor / Growth / Rates         │
    └─────┬──────────────────────────────────────┘
          │
    ┌─────▼──────────────────────────────────────┐
    │     08 Marchés (yfinance)                   │
    │  Corrélations thèmes × variations (J→J+30)  │
    │  SPY / NASDAQ / BTC                         │
    └────────────────────────────────────────────┘
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <url-du-depot>
cd dm-analyse-textuelle-fomc
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Placer le corpus

Copier les fichiers `.txt` du corpus FOMC dans le dossier `archive/` :

```
archive/
├── FOMCpresconf20200916.txt
├── FOMCpresconf20201105.txt
└── ...
```

Les fichiers doivent respecter la convention de nommage `FOMCpresconfYYYYMMDD.txt`.

---

## Utilisation

### Option A — Notebook interactif (recommandé)

```bash
jupyter lab notebooks/analyse_fomc.ipynb
```

Le notebook couvre l'intégralité de l'analyse avec des cellules de texte explicatives, les graphiques intégrés et les résultats commentés.

### Option B — Scripts Python autonomes

Chaque script peut être exécuté indépendamment :

```bash
# Pré-traitement seul
python scripts/01_preprocessing.py

# Fréquences + nuages de mots
python scripts/02_frequences.py

# N-grammes
python scripts/03_ngrammes.py

# TF-IDF par période
python scripts/04_tfidf.py

# Tonalités économiques
python scripts/05_sentiments.py

# AFC + Clustering
python scripts/06_afc_clustering.py

# Thèmes macro
python scripts/07_themes.py

# Marchés financiers
python scripts/08_marches.py
```

Chaque script sauvegarde ses figures dans `outputs/figures/` et ses tableaux dans `outputs/tables/`.

---

## Méthodologie détaillée

### Pré-traitement (script 01)

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Catégories POS retenues | Noms (N\*), Adjectifs (J\*) | Catégories les plus riches sémantiquement pour l'économie |
| Lemmatiseur | WordNetLemmatizer (NLTK) | Standard, performant sur l'anglais |
| Longueur minimale | 3 caractères | Filtrer les abréviations parasites |
| Stopwords | NLTK english + 90 termes personnalisés | Noms propres (journalistes, présidents), acronymes médias |

### Analyse de tonalités (script 05)

Approche **lexicale** (Loughran–McDonald-like) adaptée au discours de banque centrale. Quatre catégories :

| Catégorie | Exemples |
|-----------|---------|
| Optimisme | recovery, resilient, strength, momentum |
| Pessimisme | recession, decline, fragile, collapse |
| Incertitude | risk, volatility, disruption, ambiguity |
| Stabilité | stable, anchored, balanced, orderly |

### Réduction de dimension (script 06)

- **TF-IDF** : `TfidfVectorizer(max_features=10 000)`
- **SVD tronquée** : `TruncatedSVD(n_components=2)` → projection 2D (AFC)
- **K-Means** : `k=3`, données normalisées (`StandardScaler`), `n_init=10`

### Corrélations marché (script 08)

Score thématique par discours × variation de prix à horizon T, T+1, T+3, T+7, T+30 (corrélation de Pearson). Sources : `yfinance` (SPY, ^IXIC, BTC-USD).

---

## Lexiques et paramètres

Tous les lexiques (sentiments, thèmes, stopwords) et paramètres d'analyse sont centralisés dans `config.py` pour faciliter la reproductibilité et la modification.

```python
# Exemple — modifier la fenêtre de cooccurrence :
WINDOW_SIZE = 5   # → changer ici uniquement
```

---

## Résultats principaux

- **2020** : vocabulaire dominé par `supply`, `pandemic`, `support` — pic d'incertitude
- **2021–2022** : basculement vers `inflation`, `price`, `rate` — cycle de resserrement le plus agressif depuis 40 ans
- **2022–2023** : discours hawkish maximal — corrélation négative identifiée avec SPY à J+7/J+30
- **2024–2025** : normalisation progressive, retour de `growth` et `stability`

---
