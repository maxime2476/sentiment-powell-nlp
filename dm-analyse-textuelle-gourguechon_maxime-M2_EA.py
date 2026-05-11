# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:42:55 2025
@author: maxime gourguechon
"""

import os
import re
import nltk 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import yfinance as yf
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

DOSSIER_CORPUS = r"C:\Users\maxou\Documents\Cours\Analyse quantitative textuelle\Devoir\archive"
FENETRE = 5     # fenêtre de cooccurrence pour ngrammes
TOP_N = 100     # nombre max d'items à garder pour les classements
FREQ_MIN_BIGRAMME = 5    # fréquence minimum pour bigrammes
FREQ_MIN_TRIGRAMME = 3     # fréquence minimum pour trigrammes

lemmatiseur = WordNetLemmatizer()    #formule pour lemmatiser
mots_vides = set(stopwords.words("english"))      # stopword en anglais librairie

# stopwords personnalisés sous forme de dictionnaire
stop_perso = {
    'name','thank','thanks','chair','powell','michelle','smith','vice','chairman','madam',
    'sir','mr','ms','mrs','question','answer','press','conference','fomc','nick','timiraos','michael',
    'mckee','bloomberg','howard','schneider','reuters','edward','lawrence','fox','rachel','siegel','greg',
    'robb','chris','rugaber','victoria','politico','nancy','steve','liesman','cnbc','richard','escobedo',
    'jay','brien','abc','brendan','pedersen','punchbowl','jasinski','nicholas','cbs','kosuke','takami',
    'nikkei','pedersen','daniel','avis','dorsey','bryan','mena','elizabeth','schulze','amara','mena','omeokwe',
    'basel','iii','megan','cassella','gura','catarina','david','gura','matthew','torres','pedersen','jeff','cox',
    'jones','nicole','claire','anneken','mark','hamrick','saraiva','craig','kiernan','paul','jonnelle',
    'craig','would','could','say','know','think','going','got','yeah','na','uh','percent','per','cent',
    'maria','capurro','click','ann','matt','egan','kyle','campbell','evan','ryser','miller','jackson',
    'andrew','ackerman','hannah','lang','cnn','npr','simon','rabinovitch','jean','yung','scott','horsley',
    'neil','irwin','brian','cheung','boesler','jennifer','schonberger','james','mike','yahoo','thing'
}
mots_vides |= stop_perso
# retirer noms propres, médias, etc., pour garder le principal pour analyse eco

# on lit tous les .txt du dossier + dataframe
fichiers = [f for f in sorted(os.listdir(DOSSIER_CORPUS)) if f.endswith(".txt")]

enregistrements = []

# boucle sur tous les fichiers du dossier
for textes in fichiers:
    chemin = os.path.join(DOSSIER_CORPUS, textes)
    with open(chemin, encoding="utf-8") as f:
        texte = f.read()
    
    # on récupère la date présent dans le titre du texte
    motif = re.search(r'(\d{8})', textes)
    if motif:
        date = pd.to_datetime(motif.group(1), format="%Y%m%d")
        annee = date.year

    # tokenisation (minuscules + on garde que des tokens alphabétiques)
    #d'abord lemmatisation + pos puis stopword pour éviter d'enlever des infos importantes
    tokens = [w for w in word_tokenize(texte.lower(), language="english") if w.isalpha()]

    # POS
    etiquettes_pos = nltk.pos_tag(tokens)

    #lemmatisation sur seulement les noms avec N et adjectifs avec J
    # 'je me suis renseigné sur DataCamp pour créer ce code'
    lemmes = [
        lemmatiseur.lemmatize(w, pos=(wordnet.ADJ if p.startswith("J") else wordnet.NOUN))
        for w,p in etiquettes_pos
        if (p.startswith("J") or p.startswith("N")) and len(w)>=3
    ]

    #filtrer avec les stopwords
    lemmes = [w for w in lemmes if w not in mots_vides]
    
    # création du dataframe pour obtenir des infos importantes sur la structure 
    enregistrements.append({"textes": textes, "date": date, "year": annee, "tokens": len(tokens), "lemmas": lemmes, "texte_nettoye": " ".join(lemmes)})

df_discours = pd.DataFrame(enregistrements).dropna(subset=["texte_nettoye"])
# liste aplatie de tous les lemmes (pour collocations globales)
# pour cette partie j'ai demandé l'aide des IA pour créer quelque chose de claire (pour le dataframe)
lemmes_filtres = [lemma for sublist in df_discours["lemmas"] for lemma in sublist]
df_discours.info()

df_discours["tokens"].describe()

# histogramme
plt.figure(figsize=(10,6))
sns.histplot(df_discours["tokens"], bins=8, kde=True)
plt.title("Distribution du nombre de tokens par discours", fontsize=14)
plt.xlabel("Nombre de tokens")
plt.ylabel("Fréquence")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# j'ai vu sur un site ce code que j'ai décidé de mettre afin de montrer les plus longs et plus courts textes en utilisant la longeur lenght
print(df_discours[["textes", "tokens"]].sort_values(by="tokens", ascending=False).head())

print(df_discours[["textes", "tokens"]].sort_values(by="tokens", ascending=True).head())

# lineplot
plt.figure(figsize=(10,5))
sns.lineplot(data=df_discours, x="year", y="tokens", marker="o")
plt.title("Evolution du nombre de tokens par discours (FOMC)")
plt.xlabel("Année")
plt.ylabel("Nombre de tokens")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# création de la variable year regroupant les discours par année (variable year deja créer)
annees = sorted(df_discours["year"].unique())
top10_mots_par_annee = {}

# boucle pour créer chaque année un nuage de mots
for y in annees:
    # récuperation des lemmes des discours de l'année x
    lemmes_annee = [w for sub in df_discours.loc[df_discours["year"]==y, "lemmas"] for w in sub]
    total_tokens = len(lemmes_annee)
    
    #comptage des mots + calcul fréquence
    compteur = Counter(lemmes_annee)
    freq_relative = {w: c/total_tokens for w,c in compteur.items()}
    
    # création variable top10 
    top10 = sorted(freq_relative.items(), key=lambda x:x[1], reverse=True)[:10]
    top10_mots_par_annee[y] = top10
    
    # nuage de mots
    nuage = WordCloud(width=1000, height=600, background_color="white").generate_from_frequencies(freq_relative)
    plt.figure(figsize=(12,6))
    plt.imshow(nuage, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Nuage de mots – {y}")
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # pareil pour bigrammes
    if len(lemmes_annee)>=2:
        finder = BigramCollocationFinder.from_words(lemmes_annee, window_size=FENETRE)
        # on enleve les mots à moins de 3 caractères car trop court ajoute trop de biais et ceux dans la variable stop
        finder.apply_word_filter(lambda w: (len(w)<3) or (w in mots_vides))
        #comptes le nombre de bigrammes + calcul fréquence et on prends ceux qui ne sont pas égaux entre eux (w1 différent de w1)
        comptes_bigrams = { (w1, w2): c for (w1, w2), c in finder.ngram_fd.items() if w1 != w2 }
        total_bigrammes = sum(comptes_bigrams.values())
        freq_relative_bigrammes = {f"{w1}_{w2}": c/total_bigrammes for (w1,w2),c in comptes_bigrams.items()}
        #nuages de mots
        if freq_relative_bigrammes:
            nuage_bi = WordCloud(width=1000, height=600, background_color="white").generate_from_frequencies(freq_relative_bigrammes)
            plt.figure(figsize=(12,6))
            plt.imshow(nuage_bi, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Nuage de bigrammes – {y}")
            plt.tight_layout()
            plt.show()
            plt.close()

#création dataframe + visu
df_top10 = pd.DataFrame([
    {"year":y, "word":w, "freq":f, "rank":i+1}
    for y,lst in top10_mots_par_annee.items()
    for i,(w,f) in enumerate(lst)
])

print(df_top10)

# créations de deux classes bigrammes et trigrammes avec leur information, leurs mesures, leur frequence minimum et les meilleurs
for n, finder_class, measure_class, freq_minimum, top_n, etiquette in [
    (2, BigramCollocationFinder, BigramAssocMeasures, FREQ_MIN_BIGRAMME, TOP_N, "bigram"),
    (3, TrigramCollocationFinder, TrigramAssocMeasures, FREQ_MIN_TRIGRAMME, TOP_N, "trigram")
]:
    #on crée le finder sur tous les lemmes, avec une fenêtre de cooccurrence choisis à 5 de facon arbitraire
    finder = finder_class.from_words(lemmes_filtres, window_size=FENETRE)
    # on filtre en enlevanr ceux inférieur à 3 caractères
    finder.apply_word_filter(lambda w: (len(w)<3) or (w in mots_vides))
    # filtre par frequence minimum histoire d'avoir des resultats pertinents
    finder.apply_freq_filter(freq_minimum)
    # on récupère tous les ngram (ngram, fréquence)
    candidats = finder.ngram_fd.items()
    # on enlève les n-grammes avec répétition de mot
    candidats = [(ng, freq) for ng, freq in candidats if len(set(ng)) == len(ng)]
    # on trie par fréquence
    top_freq = sorted(candidats, key=lambda x: x[1], reverse=True)[:top_n]
    # dataframe propre
    df_freq = pd.DataFrame([ng for ng, freq in top_freq], 
                           columns=[f"{etiquette}_{i+1}" for i in range(n)])
    print(df_freq)

lemmes_par_periode = df_discours.groupby("year")["lemmas"].sum()
textes_par_periode = {label:" ".join(tokens) for label,tokens in lemmes_par_periode.items() if tokens}
print(lemmes_par_periode)
print(textes_par_periode)

#boucle sur bgrammes (2,2) et trigrammes (3,3)
for plage_ngram, nom_fichier in [((2,2),"2-2"), ((3,3),"3-3")]:
    # vectorizer tf-idf ngrams
    vectoriseur = TfidfVectorizer(ngram_range=plage_ngram, min_df=1, max_df=1.0, stop_words=None)
    # fit+transform sur les textes par année
    X = vectoriseur.fit_transform(textes_par_periode.values())
    #recuperation des noms
    traits = vectoriseur.get_feature_names_out()
    # filtre pour éviter les doublons
    indices_traits = [i for i, f in enumerate(traits) if len(set(f.split())) == len(f.split())]
    X = X[:, indices_traits]
    #noms de colonnes correspondant
    traits = [traits[i] for i in indices_traits]
    #liste de périodes
    noms_periodes = list(textes_par_periode.keys())
    #dataframe tf-idf année et ngrams
    df_tfidf = pd.DataFrame(X.toarray(), index=noms_periodes, columns=traits)
    
    # boucle pour chaque année 20 meilleurs ngram TF-IDF
    termes_cles = []
    for period in noms_periodes:
        sub = df_tfidf.loc[period].sort_values(ascending=False)[:20]
        for term, score in sub.items():
            termes_cles.append((period, term, score))
    df_termes_cles = pd.DataFrame(termes_cles, columns=["period","term","tfidf"])
    print(df_termes_cles)
    
    # barplot top 10 par période
    plt.figure(figsize=(11,7))
    sns.barplot(data=df_termes_cles.groupby("period").head(10), x="tfidf", y="term", hue="period", dodge=False)
    plt.title(f"Top TF-IDF par période (ngram {nom_fichier})")
    plt.tight_layout()
    plt.show()
    plt.close()

# création dictionnaires regroupant les synonymes anglais des sentiments mentionnés
sentiments = {
    "optimism": [
        "optimism", "hope", "expectation", "recovery", "expansion", "progress",
        "improve", "growth", "resilient", "strength", "strong", "solid",
        "positive", "upturn", "advance", "gain", "momentum", "enhance", "bright", "support"
    ],
    
    "pessimism": [
        "decline", "slowdown", "weakness", "recession", "contraction", "negative",
        "fragile", "downturn", "deteriorate", "drop", "fall", "loss", "unfavorable",
        "stagnant", "collapse", "crisis", "struggle", "challenge", "decrease", "pressure"
    ],
    
    "uncertainty": [
        "uncertainty", "risk", "volatility", "instability", "doubt", "concern",
        "unpredictable", "turbulent", "unknown", "fear", "shock", "unsettled",
        "ambiguity", "fluctuation", "uneven", "fragility", "hesitation", "exposure", "disruption", "speculation"
    ],
    
    "stability": [
        "stable", "anchored", "steady", "resilient", "balanced", "sustainable",
        "robust", "consistent", "sound", "firm", "secure", "enduring", "moderate",
        "controlled", "predictable", "orderly", "durable", "calm", "equilibrium", "constant"
    ]
}

# les lemmes par année + calcule fréquence relative de chaque sentiment
lemmes_par_periode = df_discours.groupby("year")["lemmas"].sum()
textes_par_periode = {label:" ".join(tokens) for label,tokens in lemmes_par_periode.items() if tokens}

# chaque année avec sentiment
scores_sentiment = []   #stockage des resultats
for label, text in textes_par_periode.items():
    words = text.split() #separation en tokens des textes
    total = len(words)  # nombre total de mots
    # boucle catégorie de sentiment
    for sent, keywords in sentiments.items():       
        freq = sum(text.count(k) for k in keywords) / total   # fréquence relative
        scores_sentiment.append((label, sent, freq))

df_sentiments = pd.DataFrame(scores_sentiment, columns=["year", "sentiment", "freq"])
print(df_sentiments)

# lineplot 
plt.figure(figsize=(12,6))
sns.lineplot(data=df_sentiments, x="year", y="freq", hue="sentiment", marker="o")
plt.title("Évolution des tonalités économiques dans les discours (Fed Press Conferences)")
plt.tight_layout()
plt.show()
plt.close()

print(df_sentiments.pivot(index="year", columns="sentiment", values="freq").round(4))

# vectorizer tf-idf
vectoriseur = TfidfVectorizer(max_features=10000)
X = vectoriseur.fit_transform(df_discours["texte_nettoye"])   # matrice de textes
#svd tronquée avec une projection à 2 dim pour interprétation
svd = TruncatedSVD(n_components=2, random_state=42)
coordonnees = svd.fit_transform(X)   #projection textes                             

#datframe 2 dimensions
df_svd = pd.DataFrame(coordonnees, columns=["Dim1","Dim2"])
df_svd["year"] = df_discours["year"]

print(df_svd)

# scatterplot afc (svd)
plt.figure(figsize=(9,7))
sns.scatterplot(data=df_svd, x="Dim1", y="Dim2", hue="year", palette="viridis", s=70)
plt.title("AFC – Projection sémantique")
plt.tight_layout()
plt.show()
plt.close()

#même chose
vectoriseur = TfidfVectorizer(max_features=10000)
X = vectoriseur.fit_transform(df_discours["texte_nettoye"])
svd = TruncatedSVD(n_components=2, random_state=42)
coordonnees = svd.fit_transform(X)

df_svd = pd.DataFrame(coordonnees, columns=["Dim1", "Dim2"])
df_svd["year"] = df_discours["year"]

plt.figure(figsize=(9, 7))
sns.scatterplot(data=df_svd, x="Dim1", y="Dim2", hue="year", palette="viridis", s=70)
plt.title("AFC – Projection sémantique des discours")

# noms utilisés et leurs poids
termes = vectoriseur.get_feature_names_out()
composantes = svd.components_  #taille 2xnb_mots
print(composantes.shape)

# mise à l'échelle
x_min, x_max = df_svd["Dim1"].min(), df_svd["Dim1"].max()
y_min, y_max = df_svd["Dim2"].min(), df_svd["Dim2"].max()

comp_x = composantes[0, :] #chargement mots dim1
comp_y = composantes[1, :] #chargement mots dim2

#normalisation sinon points en dehors du cadre sans cela
comp_x_norm = (comp_x - comp_x.min()) / (comp_x.max() - comp_x.min()) * (x_max - x_min) + x_min
comp_y_norm = (comp_y - comp_y.min()) / (comp_y.max() - comp_y.min()) * (y_max - y_min) + y_min

#calcul importance mots = somme des valeurs absolus sur les 2 dimensions et top 5
importance = np.sum(np.abs(composantes), axis=0) 
top_indices = np.argsort(importance)[-5:][::-1]

#texte des mots sur le graphique
for idx in top_indices:
    plt.text(comp_x_norm[idx], comp_y_norm[idx], termes[idx], fontsize=12, fontweight='bold', color='red')

X_normalise = StandardScaler().fit_transform(coordonnees)         # standardisation coordonnnées
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # kmeans 3 familles
df_svd["cluster"] = kmeans.fit_predict(X_normalise) # discours attitré à un cluster

# scatterplot cluster
plt.figure(figsize=(9,7))
sns.scatterplot(data=df_svd, x="Dim1", y="Dim2", hue="cluster", palette="viridis", s=70)
plt.title("Clusters K-Means (AFC)")
plt.tight_layout()
plt.show()
plt.close()

# pareil qu'avant 4 dictionnaires incluant les synonymes ou mots s'y rapprochant
themes = {
    "inflation": [
        "inflation", "price", "cost", "consumer", "wage",
        "goods", "services", "index", "increase", "spending",
        "demand", "supply", "prices"
    ],
    "labor": [
        "labor", "employment", "job", "worker", "union",
        "workforce", "hiring", "unemployment", "hours", "pay",
        "salary", "benefits", "recruitment"
    ],
    "growth": [
        "growth", "economic", "gdp", "output", "expansion",
        "investment", "productivity", "development", "recovery",
        "progress", "increase", "activity"
    ],
    "rates": [
        "rate", "interest", "borrowing", "lending", "yield",
        "policy", "benchmark", "credit", "loan", "return",
        "cost", "discount", "spread"
    ]
}

scores_theme = [] #stockage
for theme, keywords in themes.items(): #boucle sur thème
    for year, text in textes_par_periode.items(): # boucle sur chauqe année
        freq = sum(text.count(k) for k in keywords) / len(text.split()) # fréquence relative
        scores_theme.append((year, theme, freq))

df_themes = pd.DataFrame(scores_theme, columns=["year", "theme", "freq"])

# lineplot
plt.figure(figsize=(12,6))
sns.lineplot(data=df_themes, x="year", y="freq", hue="theme", marker="o")
plt.title("Évolution thématique : inflation, travail, croissance et taux (fréquence relative)")
plt.tight_layout()
plt.show()
plt.close()

# cette prochaine partie a été fait en grande partie par l'IA faute de compétences trop poussés en python

df_discours = df_discours.copy()  # copie de sécurité
df_discours[['Dim1','Dim2']] = df_svd[['Dim1','Dim2']]  #coords SVD sur chaque discours
df_discours['date'] = pd.to_datetime(df_discours['date'])  # problème de concordance avec date de la librarie yfinance

for theme, keywords in themes.items():  # boucle sur chaque thème
    df_discours[theme+'_score'] = df_discours['texte_nettoye'].apply(  #création d’une colonne score par thème
        lambda txt: sum(txt.count(k) for k in keywords)/len(txt.split())  # part des mots du thème dans le texte
    )

horizons = ['pct_change','pct_change+1','pct_change+3','pct_change+7','pct_change+30']  #horizons de perfs marché (T, T+1, etc.)

tickers_actifs = {"SPY":"SPY", "BTC":"BTC-USD", "NASDAQ":"^IXIC"}  # mapping actifs -> ticker yfinance
dfs_marches = {}  # dict pour stocker les DataFrames des marchés téléchargés

for nom_actif, ticker in tickers_actifs.items():  # boucle sur chaque actif
    df_prix = yf.Ticker(ticker).history(period="5y", auto_adjust=True)  #historique 5 ans et prix ajustés
    df_prix.index = pd.to_datetime(df_prix.index).tz_localize(None)  #index datetime sans timezone
    
    #variations en % le jour même et en décalé (T+1, +3, +7, +30)
    df_prix['pct_change']   = df_prix['Close'].pct_change() * 100  # variation journalière en %
    df_prix['pct_change+1'] = df_prix['Close'].pct_change(periods=1).shift(-1) * 100  #variation sur 1 jour 
    df_prix['pct_change+3'] = df_prix['Close'].pct_change(periods=3).shift(-3) * 100  #variation sur 3 jours
    df_prix['pct_change+7'] = df_prix['Close'].pct_change(periods=7).shift(-7) * 100  #variation sur 7 jours
    df_prix['pct_change+30']= df_prix['Close'].pct_change(periods=30).shift(-30) * 100 # variation sur 30 jours
    
    dfs_marches[nom_actif] = df_prix  #on range le DF dans le dict par nom d’actif

corrs_themes_tous = {}  # contiendra toutes les matrices de corrélations pour les différents actifs

for nom_actif, df_marche in dfs_marches.items():
    df_fusion = pd.merge(df_discours, df_marche, left_on='date', right_index=True, how='left')  # left join sur la date (gauche = discours)
    
    corrs_themes = {}  #corrélations par thème (dict)
    for theme in themes.keys():  #boucle thèmes
        col = theme+'_score'  #nom de la colonne du score thème
        if col in df_fusion.columns:
            corrs_themes[theme] = df_fusion[[col]+horizons].corr()[col][horizons]  # corr(score, variations) pour chaque horizon
    
    df_corr = pd.DataFrame(corrs_themes)
    corrs_themes_tous[nom_actif] = df_corr  # on stocke la matrice pour cet actif
    print(df_corr)
    
    #heatmap
    plt.figure(figsize=(8,6))  # taille de la figure
    sns.heatmap(df_corr.T, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Corrélations thèmes – {nom_actif} (différents horizons)") 
    plt.xlabel("Horizon")
    plt.ylabel("Thème") 
    plt.tight_layout()
    plt.show()
    plt.close()  

#deja fait
X_normalise = StandardScaler().fit_transform(df_discours[['Dim1','Dim2']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  
df_discours['cluster'] = kmeans.fit_predict(X_normalise)

print(df_discours[['textes','year','cluster']].head())

textes_cluster = df_discours.groupby('cluster')['texte_nettoye'].apply(lambda x: " ".join(x))  #texte par cluster
tfidf = TfidfVectorizer(max_features=10000, stop_words="english")  # TF-IDF limité à 10000 termes et avec stopwords
X = tfidf.fit_transform(textes_cluster)

termes = tfidf.get_feature_names_out()  #liste de termes conservés
df_tfidf_cluster = pd.DataFrame(X.toarray(), index=textes_cluster.index, columns=termes)

#top mots par cluster
print(df_tfidf_cluster.T.sort_values(by=0, ascending=False).head(10))
print(df_tfidf_cluster.T.sort_values(by=1, ascending=False).head(10))
print(df_tfidf_cluster.T.sort_values(by=2, ascending=False).head(10))

mots_hawkish = ["inflation","rate","interest","tightening","price","cost","increase"]  #lexique “hawkish”
mots_dovish  = ["growth","employment","stimulus","support","recovery","investment"]  #lexique “dovish”

scores_cluster = {}  # dictionnaire stocke score hawkish/dovish par cluster
for cluster, row in df_tfidf_cluster.iterrows():  # boucle sur chaque cluster
    hawk_score = row[[w for w in row.index if w in mots_hawkish]].sum()  #somme TF-IDF des mots hawkish
    dov_score  = row[[w for w in row.index if w in mots_dovish]].sum()  #somme TF-IDF des mots dovish
    scores_cluster[cluster] = {"hawkish": hawk_score, "dovish": dov_score}

df_cluster_scores = pd.DataFrame(scores_cluster).T #transposer plus faiclement interpretable
print(df_cluster_scores)

horizons = ['pct_change','pct_change+1','pct_change+3','pct_change+7','pct_change+30']
df_discours = df_discours.merge(df_marche[horizons], left_on='date', right_index=True, how='left')  # left join avec df_marche

reaction_marche_par_cluster = {}  # stocke la moyenne des variations par cluster
for cluster in df_discours['cluster'].unique():  #boucle sur les clusters
    sous_ensemble = df_discours[df_discours['cluster']==cluster]  #sous-ensemble pour ce cluster
    reaction_marche_par_cluster[cluster] = sous_ensemble[horizons].mean()  # moyenne simple des marchés

df_reaction_marche_cluster = pd.DataFrame(reaction_marche_par_cluster).T
df_reaction_marche_cluster = df_reaction_marche_cluster.sort_index()
print(df_reaction_marche_cluster)
