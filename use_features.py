# use_features.py — exploiter features_tfidf.csv (hawk - dove)
# Entrées attendues:
#   - features_tfidf.csv : doc_id, tfidf_hawk_minus_dove, tfidf_hawk_minus_dove_z
#   - ecb_text_corpus.csv (optionnel) : métadonnées (doc_id, date_time, channel, title, speaker, url, source_file)

import pandas as pd
import numpy as np
from pathlib import Path

FEAT = "features_tfidf.csv"
META = "ecb_text_corpus.csv"   # présent chez toi d'après la capture

# 1) Charger
f = pd.read_csv(FEAT, dtype={"doc_id": str})
if "tfidf_hawk_minus_dove_z" not in f.columns:
    x = f["tfidf_hawk_minus_dove"]
    f["tfidf_hawk_minus_dove_z"] = (x - x.mean()) / x.std(ddof=0)

# 2) Joindre les métadonnées si dispo
base_cols = ["doc_id","date_time","channel","title","speaker","url","source_file"]
if Path(META).exists():
    m = pd.read_csv(META, parse_dates=["date_time"], dtype={"doc_id": str}, low_memory=False)
    m = m[[c for c in base_cols if c in m.columns]]
    df = m.merge(f, on="doc_id", how="inner")
else:
    df = f.copy()

# 3) Z-score par canal (recommandé pour comparer speech / press_conference / qna)
def z(s): 
    std = s.std(ddof=0)
    return (s - s.mean()) / (std if std != 0 else 1.0)

if "channel" in df.columns:
    df["tfidf_hmd_z_ch"] = df.groupby("channel")["tfidf_hawk_minus_dove"].transform(z)
else:
    df["tfidf_hmd_z_ch"] = df["tfidf_hawk_minus_dove_z"]

# 4) Labels (dovish / neutral / hawkish) selon le z-score par canal
df["lex_label"] = pd.cut(
    df["tfidf_hmd_z_ch"], bins=[-np.inf, -0.67, 0.67, np.inf],
    labels=["dovish","neutral","hawkish"]
)

# 5) Exports utiles
df.to_csv("features_ready.csv", index=False)

# Tops / bas
df.sort_values("tfidf_hmd_z_ch", ascending=False).head(20)\
  .to_csv("top20_hawkish.csv", index=False)
df.sort_values("tfidf_hmd_z_ch", ascending=True).head(20)\
  .to_csv("bottom20_dovish.csv", index=False)

# Séries et stats si date_time / channel disponibles
if "date_time" in df.columns:
    (df.set_index("date_time").resample("M")["tfidf_hmd_z_ch"].mean()
       .rename("mean_z").to_csv("series_hawk_dove_monthly.csv"))
if "channel" in df.columns:
    (df.groupby("channel")["tfidf_hmd_z_ch"].agg(["mean","median","std","count"])
       .to_csv("by_channel_stats.csv"))

# 6) Petit résumé console
print("✅ Écrit: features_ready.csv")
print("📄 Aussi: top20_hawkish.csv, bottom20_dovish.csv")
if "date_time" in df.columns: print("📈 series_hawk_dove_monthly.csv")
if "channel" in df.columns:   print("📊 by_channel_stats.csv")
print("\nAperçu:")
cols_show = [c for c in ["date_time","channel","speaker","title",
                         "tfidf_hawk_minus_dove","tfidf_hmd_z_ch","lex_label"] if c in df.columns]
print(df[cols_show].head(5))
