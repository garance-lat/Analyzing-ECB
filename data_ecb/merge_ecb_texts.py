# merge_ecb_texts.py
from __future__ import annotations
import argparse, re
from pathlib import Path
from itertools import product
import pandas as pd

REQ_COLS = ["date","title","link","text"]

SYNONYMS = {
    "date": {"date","publication_date","pub_date","time","datetime"},
    "title": {"title","headline","subject"},
    "link": {"link","url","urls","href"},
    "text": {"text","content","body","speech","transcript"},
}

def clean_text_basic(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"http\S+", " ", s)
    return s.strip()

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # lower/strip
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    # synonym mapping
    cols = set(df.columns)
    rename_map = {}
    for target, alts in SYNONYMS.items():
        for a in alts:
            if a in cols:
                rename_map[a] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    # ensure required columns exist
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = None
    return df

def smart_read_csv(path: Path, required=tuple(REQ_COLS), debug=False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    encodings = ["utf-8-sig","utf-8","cp1252","latin-1"]
    seps = [",",";","\t","|"]
    last_err = None
    for enc, sep in product(encodings, seps):
        try:
            df = pd.read_csv(
                path, encoding=enc, sep=sep, engine="python",
                on_bad_lines="skip", encoding_errors="ignore"
            )
            df = normalize_headers(df)
            # colonnes OK ?
            if set(required).issubset(df.columns):
                if debug:
                    print(f"[smart_read_csv] OK enc={enc} sep={repr(sep)} cols={list(df.columns)[:8]}...")
                return df
            else:
                if debug:
                    print(f"[smart_read_csv] enc={enc} sep={repr(sep)} -> colonnes insuffisantes: {df.columns.tolist()}")
        except Exception as e:
            last_err = e
            if debug:
                print(f"[smart_read_csv] FAIL enc={enc} sep={repr(sep)} err={e}")
            continue
    raise RuntimeError(f"Impossible de lire {path} (encodages testés={encodings}, séparateurs testés={seps}). Dernière erreur: {last_err}")

def prepare_block(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    out = df.copy()
    # parse dates (tolérant)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    out["title"] = out["title"].fillna("")
    out["link"]  = out["link"].fillna("")
    out["text"]  = out["text"].fillna("").map(clean_text_basic)
    out["source_type"] = source_type
    out = out.dropna(subset=["date"])
    return out[["date","title","link","text","source_type"]]

def main():
    ap = argparse.ArgumentParser(description="Merge ECB speeches and press conferences into one CSV")
    ap.add_argument("--speeches", type=Path, required=True)
    ap.add_argument("--pressers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        print(f"[debug] speeches={args.speeches}")
        print(f"[debug] pressers={args.pressers}")
        print(f"[debug] out={args.out}")

    df_s = smart_read_csv(args.speeches, debug=args.debug)
    df_p = smart_read_csv(args.pressers, debug=args.debug)

    if args.debug:
        print("Speeches columns:", df_s.columns.tolist())
        print("Pressers columns:", df_p.columns.tolist())

    sp = prepare_block(df_s, "speech")
    pr = prepare_block(df_p, "press_conf")

    merged = pd.concat([sp, pr], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"✅ Merged {len(merged)} rows -> {args.out}")

if __name__ == "__main__":
    main()
