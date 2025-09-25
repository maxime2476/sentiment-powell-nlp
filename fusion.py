# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:09:35 2025

@author: maxou
"""

import os
import re
from os import listdir
from os.path import isfile, join, splitext

rep = r"C:\Users\maxou\Documents\Cours\Analyse quantitative textuelle\Devoir\archive"

textes = [f for f in listdir(rep) if isfile(join(rep, f)) and f.lower().endswith(".txt")]
textes.sort()

os.chdir(rep)

tous = ""
for element in textes:
    f = open(element, "r", encoding="utf-8")
    textebrut = f.read()
    f.close()
    
    pat = re.compile(r'<\s*NAME\s*>\s*(.*?)\s*<\s*/\s*NAME\s*>', re.IGNORECASE)
    tags = list(pat.finditer(textebrut))
    
    if tags:
        keep = []
        for i, m in enumerate(tags):
            who = m.group(1).strip().upper()
            seg_start = m.end()
            seg_end = tags[i+1].start() if i+1 < len(tags) else len(textebrut)
            if who == "CHAIR POWELL":
            # on garde le tag <NAME>CHAIR POWELL</NAME> + tout ce qui suit
            # jusqu'au prochain <NAME> (qui sera supprimé s'il n'est pas Powell)
                keep.append(textebrut[seg_start:seg_end])
        textebrut = "".join(keep)
    else:
    # S'il n'y a aucune balise NAME, on peut vider le texte
    # (ou le laisser tel quel si tu préfères)
        textebrut = ""

    t = textebrut.replace("\n", " ")
    t = t.replace("-", " ")
    t = t.replace(";", " ")

    titre = splitext(element)[0]
    sep = "****" + titre

    tous = tous + sep + "\n\n" + t.strip() + "\n\n"

dossier = r"C:\Users\maxou\Documents\Cours\Analyse quantitative textuelle\Devoir"
sortie = "fusion_powell.txt"

with open(os.path.join(dossier, sortie), "w", encoding="utf-8") as f_out:
    f_out.write(tous)


        
