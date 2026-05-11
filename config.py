# -*- coding: utf-8 -*-
"""
config.py — Configuration centrale du projet
Analyse Textuelle des Conférences de Presse du FOMC (2020-2025)
Maxime Gourguechon — M2 Économie Appliquée
"""

import os

# Évite le warning KMeans/MKL sur Windows (doit être défini avant tout import sklearn)
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "archive")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR  = os.path.join(OUTPUTS_DIR, "tables")

# Création automatique des dossiers de sortie
for _dir in (FIGURES_DIR, TABLES_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Paramètres d'analyse
# ---------------------------------------------------------------------------
WINDOW_SIZE        = 5   # fenêtre de cooccurrence pour les n-grammes
TOP_N              = 100  # nombre max d'items dans les classements
MIN_FREQ_BIGRAM    = 5    # fréquence minimum pour retenir un bigramme
MIN_FREQ_TRIGRAM   = 3    # fréquence minimum pour retenir un trigramme
N_CLUSTERS         = 3    # nombre de clusters K-Means
SVD_COMPONENTS     = 2    # dimensions de projection SVD
TFIDF_MAX_FEATURES = 10000
MIN_WORD_LEN       = 3    # longueur minimale des tokens conservés

# ---------------------------------------------------------------------------
# Stopwords personnalisés — noms propres, journalistes, acronymes parasites
# ---------------------------------------------------------------------------
CUSTOM_STOPWORDS = {
    'name','thank','thanks','chair','powell','michelle','smith','vice','chairman',
    'madam','sir','mr','ms','mrs','question','answer','press','conference','fomc',
    'nick','timiraos','michael','mckee','bloomberg','howard','schneider','reuters',
    'edward','lawrence','fox','rachel','siegel','greg','robb','chris','rugaber',
    'victoria','politico','nancy','steve','liesman','cnbc','richard','escobedo',
    'jay','brien','abc','brendan','pedersen','punchbowl','jasinski','nicholas',
    'cbs','kosuke','takami','nikkei','daniel','avis','dorsey','bryan','mena',
    'elizabeth','schulze','amara','omeokwe','basel','iii','megan','cassella',
    'gura','catarina','david','matthew','torres','jeff','cox','jones','nicole',
    'claire','anneken','mark','hamrick','saraiva','craig','kiernan','paul',
    'jonnelle','would','could','say','know','think','going','got','yeah','na',
    'uh','percent','per','cent','maria','capurro','click','ann','matt','egan',
    'kyle','campbell','evan','ryser','miller','jackson','andrew','ackerman',
    'hannah','lang','cnn','npr','simon','rabinovitch','jean','yung','scott',
    'horsley','neil','irwin','brian','cheung','boesler','jennifer','schonberger',
    'james','mike','yahoo','thing',
    'colby','york',
}

# ---------------------------------------------------------------------------
# Lexiques de tonalité économique
# ---------------------------------------------------------------------------
SENTIMENTS = {
    "optimism": [
        "optimism","hope","expectation","recovery","expansion","progress",
        "improve","growth","resilient","strength","strong","solid","positive",
        "upturn","advance","gain","momentum","enhance","bright","support",
    ],
    "pessimism": [
        "decline","slowdown","weakness","recession","contraction","negative",
        "fragile","downturn","deteriorate","drop","fall","loss","unfavorable",
        "stagnant","collapse","crisis","struggle","challenge","decrease","pressure",
    ],
    "uncertainty": [
        "uncertainty","risk","volatility","instability","doubt","concern",
        "unpredictable","turbulent","unknown","fear","shock","unsettled",
        "ambiguity","fluctuation","uneven","fragility","hesitation","exposure",
        "disruption","speculation",
    ],
    "stability": [
        "stable","anchored","steady","resilient","balanced","sustainable",
        "robust","consistent","sound","firm","secure","enduring","moderate",
        "controlled","predictable","orderly","durable","calm","equilibrium",
        "constant",
    ],
}

# ---------------------------------------------------------------------------
# Lexiques thématiques macro-économiques
# ---------------------------------------------------------------------------
THEMES = {
    "inflation": [
        "inflation","price","cost","consumer","wage","goods","services",
        "index","increase","spending","demand","supply","prices",
    ],
    "labor": [
        "labor","employment","job","worker","union","workforce","hiring",
        "unemployment","hours","pay","salary","benefits","recruitment",
    ],
    "growth": [
        "growth","economic","gdp","output","expansion","investment",
        "productivity","development","recovery","progress","increase","activity",
    ],
    "rates": [
        "rate","interest","borrowing","lending","yield","policy","benchmark",
        "credit","loan","return","cost","discount","spread",
    ],
}

# ---------------------------------------------------------------------------
# Actifs financiers (yfinance)
# ---------------------------------------------------------------------------
TICKERS = {
    "SPY":    "SPY",
    "NASDAQ": "^IXIC",
    "BTC":    "BTC-USD",
}
MARKET_PERIOD = "5y"

HORIZONS = [
    "pct_change",
    "pct_change+1",
    "pct_change+3",
    "pct_change+7",
    "pct_change+30",
]

# Lexiques hawkish / dovish pour l'interprétation des clusters
HAWKISH_WORDS = ["inflation","rate","interest","tightening","price","cost","increase"]
DOVISH_WORDS  = ["growth","employment","stimulus","support","recovery","investment"]
