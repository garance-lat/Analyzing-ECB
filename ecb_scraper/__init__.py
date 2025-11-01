#Copyright (c) 2024 Thomas Kientz
from .scraper import load_ecb_conferences

import pandas as pd
df = pd.read_csv("C:\Users\Garance Latieule\analyzing ECB\Analyzing-ECB\data_ecb\all_ECB_speeches.csv", sep="|", encoding="utf-8-sig")  # adapte sep/encoding si besoin
# Vérifie les colonnes si doute :
print(df.columns)