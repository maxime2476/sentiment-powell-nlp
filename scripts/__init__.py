# -*- coding: utf-8 -*-
"""
scripts/__init__.py
===================
Expose les modules numérotés (01_preprocessing.py, etc.) sous des alias
Python valides afin de permettre les imports du style :

    from scripts.preprocessing import charger_corpus
    from scripts.frequences import plot_distribution_tokens
    ...

Les noms numérotés sont conservés pour la lisibilité dans l'explorateur
de fichiers, mais Python n'autorise pas les identifiants débutant par un
chiffre — d'où cet alias via importlib.
"""

import importlib.util
import os
import sys

_PKG_DIR = os.path.dirname(__file__)

_ALIASES = {
    "preprocessing":  "01_preprocessing",
    "frequences":     "02_frequences",
    "ngrammes":       "03_ngrammes",
    "tfidf":          "04_tfidf",
    "sentiments":     "05_sentiments",
    "afc_clustering": "06_afc_clustering",
    "themes":         "07_themes",
    "marches":        "08_marches",
}

for _alias, _filename in _ALIASES.items():
    _fqn  = f"scripts.{_alias}"
    _path = os.path.join(_PKG_DIR, f"{_filename}.py")

    if _fqn not in sys.modules:
        _spec   = importlib.util.spec_from_file_location(_fqn, _path)
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_fqn] = _module
        _spec.loader.exec_module(_module)
