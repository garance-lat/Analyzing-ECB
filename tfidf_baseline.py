#!/usr/bin/env python3
# tfidf_baseline.py
# Usage:
#   python tfidf_baseline.py ecb_text_corpus.csv -o features_tfidf.csv

import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_csv")
    ap.add_argument("-o","--output", default="features_tfidf.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.corpus_csv)
    texts = df["text_clean"].fillna("").astype(str).tolist()

    # TF-IDF unigrams+bigrams (ajuste min_df si besoin)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9)
    X = vec.fit_transform(texts)
    vocab = vec.vocabulary_

    # Seeds (à adapter pour la BCE)
    seeds_pos = ["hike","tightening","restrictive","inflation","above target","increase rates"]
    seeds_neg = ["cut","easing","accommodative","disinflation","below target","decrease rates"]

    pos_idx = [vocab[w] for w in seeds_pos if w in vocab]
    neg_idx = [vocab[w] for w in seeds_neg if w in vocab]

    pos_score = np.asarray(X[:, pos_idx].mean(axis=1)).ravel() if pos_idx else np.zeros(X.shape[0])
    neg_score = np.asarray(X[:, neg_idx].mean(axis=1)).ravel() if neg_idx else np.zeros(X.shape[0])
    raw = pos_score - neg_score

    # z-score pour comparabilité
    z = (raw - raw.mean()) / (raw.std(ddof=0) + 1e-8)

    out = pd.DataFrame({
        "doc_id": df["doc_id"],
        "tfidf_hawk_minus_dove": raw,
        "tfidf_hawk_minus_dove_z": z
    })
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"✅ écrit: {args.output} | seeds capturés -> +{len(pos_idx)} / -{len(neg_idx)}")

if __name__ == "__main__":
    main()
