# analyze_features.py — exploite features_tfidf.csv (+ meta si dispo)

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

FEAT = "features_tfidf.csv"   # doc_id, tfidf_hawk_minus_dove, tfidf_hawk_minus_dove_z
META = "ecb_text_corpus.csv"  # optionnel : métadonnées + text_clean

# 1) Charger features
f = pd.read_csv(FEAT, dtype={"doc_id": str})
if "tfidf_hawk_minus_dove_z" not in f.columns:
    x = f["tfidf_hawk_minus_dove"]
    f["tfidf_hawk_minus_dove_z"] = (x - x.mean()) / x.std(ddof=0)

# 2) Joindre métadonnées si présentes
keep_meta = ["doc_id","date_time","channel","title","speaker","url","source_file","text_clean"]
if Path(META).exists():
    m = pd.read_csv(META, parse_dates=["date_time"], dtype={"doc_id": str}, low_memory=False)
    m = m[[c for c in keep_meta if c in m.columns]]
    df = m.merge(f, on="doc_id", how="inner")
else:
    df = f.copy()
    df["channel"] = df.get("channel", "unknown")

# 3) Z-score par canal + labels par quantiles (30/40/30) -> équilibre
def z_by_group(s):
    std = s.std(ddof=0)
    return (s - s.mean()) / (std if std != 0 else 1.0)

df["tfidf_hmd_z_ch"] = df.groupby("channel")["tfidf_hawk_minus_dove"].transform(z_by_group)

def label_by_quantiles(g, lo=0.30, hi=0.70):
    ql, qh = g["tfidf_hmd_z_ch"].quantile([lo, hi])
    x = g["tfidf_hmd_z_ch"]
    lbl = np.where(x <= ql, "dovish", np.where(x >= qh, "hawkish", "neutral"))
    return pd.Series(lbl, index=g.index)

df["lex_label"] = df.groupby("channel", group_keys=False).apply(label_by_quantiles)

# 4) Exports principaux
df.to_csv("features_ready.csv", index=False)
df.sort_values("tfidf_hmd_z_ch", ascending=False).head(50)\
  .to_csv("top50_hawkish.csv", index=False)
df.sort_values("tfidf_hmd_z_ch", ascending=True).head(50)\
  .to_csv("top50_dovish.csv", index=False)

# Séries/statistiques si date_time / channel dispo
if "date_time" in df.columns:
    (df.set_index("date_time").resample("M")["tfidf_hmd_z_ch"].mean()
       .rename("mean_z").to_csv("series_hawk_dove_monthly.csv"))
if "channel" in df.columns:
    (df.groupby("channel")["tfidf_hmd_z_ch"].agg(["mean","median","std","count"])
       .to_csv("by_channel_stats.csv"))

# 5) Pack de lecture (extraits)
if "text_clean" in df.columns:
    def excerpt(s, n=500):
        return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, n)
    (df.sort_values("tfidf_hmd_z_ch", ascending=False)
       .head(30)[["date_time","channel","speaker","title","url","tfidf_hmd_z_ch","text_clean"]]
       .assign(excerpt=lambda d: excerpt(d["text_clean"]))
       .drop(columns=["text_clean"])
       .to_csv("reading_pack_hawkish.csv", index=False))
    (df.sort_values("tfidf_hmd_z_ch", ascending=True)
       .head(30)[["date_time","channel","speaker","title","url","tfidf_hmd_z_ch","text_clean"]]
       .assign(excerpt=lambda d: excerpt(d["text_clean"]))
       .drop(columns=["text_clean"])
       .to_csv("reading_pack_dovish.csv", index=False))

# 6) Graphiques (matplotlib pur, un plot par figure, pas de couleurs custom)
# Histogramme global des z (par canal)
plt.figure()
df["tfidf_hmd_z_ch"].hist(bins=50)
plt.title("Distribution du score hawk−dove (z par canal)")
plt.xlabel("z-score par canal")
plt.ylabel("Nombre de documents")
plt.tight_layout()
plt.savefig("plot_z_hist.png", dpi=150)
plt.close()

# Boxplot par canal
if "channel" in df.columns:
    plt.figure()
    df.boxplot(column="tfidf_hmd_z_ch", by="channel", vert=True)
    plt.title("Score hawk−dove par canal")
    plt.suptitle("")
    plt.ylabel("z-score par canal")
    plt.tight_layout()
    plt.savefig("plot_z_by_channel.png", dpi=150)
    plt.close()

# Série mensuelle
if "date_time" in df.columns:
    plt.figure()
    (df.set_index("date_time").resample("M")["tfidf_hmd_z_ch"].mean()).plot()
    plt.title("Moyenne mensuelle du score hawk−dove (z par canal)")
    plt.xlabel("Date")
    plt.ylabel("z moyen")
    plt.tight_layout()
    plt.savefig("plot_monthly_mean.png", dpi=150)
    plt.close()

# 7) Résumé console
print("✅ Écrit : features_ready.csv")
print("📄 top50_hawkish.csv / top50_dovish.csv / by_channel_stats.csv / series_hawk_dove_monthly.csv (si colonnes)")
print("🖼  plot_z_hist.png / plot_z_by_channel.png / plot_monthly_mean.png (si colonnes)")
print("\nAperçu :")
cols = [c for c in ["date_time","channel","speaker","title","tfidf_hawk_minus_dove","tfidf_hmd_z_ch","lex_label"] if c in df.columns]
print(df[cols].head(5))
