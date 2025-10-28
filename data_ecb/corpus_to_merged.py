
# src/corpus_to_merged.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Convert normalized corpus to pipeline 'merged_texts.csv' format.")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="ecb_text_corpus.csv from ingest_normalize_ecb")
    ap.add_argument("--out", type=Path, required=True, help="Output merged_texts.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["date_time"])
    out = pd.DataFrame()
    out["date"] = df["date_time"].dt.date
    out["title"] = df.get("title","")
    out["link"] = df.get("url","")
    out["text"] = df.get("text_clean","")
    def map_src(ch: str) -> str:
        ch = str(ch).lower()
        if "speech" in ch:
            return "speech"
        else:
            return "press_conf"
    out["source_type"] = df["channel"].map(map_src)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✅ Wrote {len(out)} rows -> {args.out}")

if __name__ == "__main__":
    main()
