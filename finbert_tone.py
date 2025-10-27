#!/usr/bin/env python3
# finbert_tone.py
# Usage:
#   python finbert_tone.py ecb_text_corpus.csv -o features_finbert.csv

import argparse, sys
import pandas as pd
import numpy as np

def try_finbert():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        nlp = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, truncation=True)
        return nlp
    except Exception as e:
        print("[WARN] FinBERT indisponible -> fallback=0.0", file=sys.stderr)
        return None

def finbert_doc_score(nlp, text: str, max_chars=4500) -> float:
    if nlp is None or not isinstance(text, str) or not text.strip():
        return 0.0
    scores = []
    for i in range(0, len(text), max_chars):
        chunk = text[i:i+max_chars]
        out = nlp(chunk[:max_chars])[0]  # {'label': 'Positive/Negative/Neutral', 'score': ...}
        label = out["label"].lower()
        score = float(out["score"])
        val = (1.0 if "positive" in label else (-1.0 if "negative" in label else 0.0)) * score
        scores.append(val)
    return float(np.median(scores)) if scores else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_csv")
    ap.add_argument("-o","--output", default="features_finbert.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.corpus_csv)
    texts = df["text_clean"].fillna("").astype(str).tolist()

    nlp = try_finbert()
    scores = [finbert_doc_score(nlp, t) for t in texts]

    z = (np.array(scores) - np.mean(scores)) / (np.std(scores) + 1e-8)

    out = pd.DataFrame({
        "doc_id": df["doc_id"],
        "tone_finbert": scores,
        "tone_finbert_z": z
    })
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"✅ écrit: {args.output} (FinBERT{' OK' if nlp else ' fallback=0'})")

if __name__ == "__main__":
    main()
