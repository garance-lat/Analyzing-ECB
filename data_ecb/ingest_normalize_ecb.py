
# src/ingest_normalize_ecb.py
from __future__ import annotations
import argparse, csv, io, re, json, hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd

def robust_read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = p.read_text(encoding=enc, errors="replace")
            break
        except Exception:
            continue
    if text is None:
        raise RuntimeError(f"Cannot decode {path} with common encodings")
    head = text[:100_000]
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=";,|\t,")
        delim = dialect.delimiter
    except Exception:
        delim = max([",",";","|","\t"], key=lambda d: head.count(d))
    df = pd.read_csv(io.StringIO(text), delimiter=delim, engine="python", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    return df

def pick_col(df: pd.DataFrame, needles: List[str]) -> Optional[str]:
    cols = [c for c in df.columns if any(n in c.lower() for n in needles)]
    return cols[0] if cols else None

def parse_best_date(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["date","time","year","day","pub"])]
    best, best_ok, best_series = None, -1.0, None
    for c in candidates:
        s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True, utc=False, dayfirst=True)
        ok = float(s.notna().mean())
        if ok > best_ok:
            best, best_ok, best_series = c, ok, s
    return best, best_series

def clean_text(x) -> str:
    if pd.isna(x): return ""
    t = str(x)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("\xa0", " ")
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def stable_id(*parts) -> str:
    import hashlib
    raw = "||".join([str(p) for p in parts if p is not None])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def guess_channel_from_name(filename: str, default: str) -> str:
    f = filename.lower()
    if "conf" in f or "press" in f: return "press_conference"
    if "q&a" in f or "qa" in f:     return "qna"
    if "interview" in f:            return "interview"
    if "speech" in f:               return "speech"
    return default

def normalize_df(df: pd.DataFrame, source_name: str, fallback_channel: str) -> tuple[pd.DataFrame, Dict[str,str]]:
    title_col   = pick_col(df, ["title","subject","headline"])
    text_col    = pick_col(df, ["text","content","speech","remarks","transcript","body","contents"])
    speaker_col = pick_col(df, ["speaker","author","name","president","speakers"])
    url_col     = pick_col(df, ["url","link","href"])
    date_col, parsed_dates = parse_best_date(df)

    if parsed_dates is None and url_col is not None:
        url = df[url_col].astype(str)
        d1 = url.str.extract(r"(?P<d>\d{4}-\d{2}-\d{2})")["d"]
        d2 = url.str.extract(r"is(?P<d2>\d{8})")["d2"]
        d2 = pd.to_datetime(d2, errors="coerce", format="%Y%m%d")
        parsed_dates = pd.to_datetime(d1, errors="coerce").fillna(d2)

    channel = guess_channel_from_name(source_name, fallback_channel)

    out = pd.DataFrame()
    out["date_time"]   = parsed_dates
    out["channel"]     = channel
    out["title"]       = df[title_col] if title_col else ""
    out["speaker"]     = df[speaker_col] if speaker_col else ""
    out["role"]        = ""
    out["language"]    = ""
    out["url"]         = df[url_col] if url_col else ""
    out["text_clean"]  = (df[text_col].apply(clean_text) if text_col else "")

    out["source_file"] = source_name
    out["doc_id"] = [
        stable_id(source_name, out["title"].iloc[i], out["date_time"].iloc[i],
                  out["speaker"].iloc[i], out["text_clean"].iloc[i][:160])
        for i in range(len(out))
    ]
    out = out[(out["text_clean"].str.len()>0) & out["date_time"].notna()].copy()
    out = out.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    out["n_chars"]     = out["text_clean"].str.len()
    out["n_tokens_ws"] = out["text_clean"].str.split().apply(len)

    picked = {
        "title_col": title_col or "",
        "text_col": text_col or "",
        "speaker_col": speaker_col or "",
        "url_col": url_col or "",
        "date_col": date_col or "",
        "channel": channel,
        "source": source_name
    }
    return out, picked

def make_qa(df: pd.DataFrame, picked_a, picked_b):
    return {
        "rows_total": int(len(df)),
        "time_range": {"min": str(df["date_time"].min()) if len(df) else None,
                       "max": str(df["date_time"].max()) if len(df) else None},
        "by_channel": {str(k): int(v) for k,v in df["channel"].value_counts().items()},
        "missing_title_%": round(df["title"].astype(str).str.strip().eq("").mean()*100,2),
        "missing_speaker_%": round(df["speaker"].astype(str).str.strip().eq("").mean()*100,2),
        "median_chars": int(df["n_chars"].median() if len(df) else 0),
        "median_tokens_ws": int(df["n_tokens_ws"].median() if len(df) else 0),
        "columns_picked": {"A": picked_a, "B": picked_b}
    }

def main():
    ap = argparse.ArgumentParser(description="Ingestion & normalisation BCE (2 CSV -> corpus propre).")
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("-o","--output", default="outputs/ecb_text_corpus.csv")
    ap.add_argument("--qa", default="outputs/ecb_text_corpus_QA.json")
    args = ap.parse_args()

    df_a = robust_read_table(args.csv_a)
    df_b = robust_read_table(args.csv_b)
    norm_a, picked_a = normalize_df(df_a, Path(args.csv_a).name, fallback_channel="speech")
    norm_b, picked_b = normalize_df(df_b, Path(args.csv_b).name, fallback_channel="press_conference")

    corpus = pd.concat([norm_a, norm_b], ignore_index=True).sort_values("date_time")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(args.output, index=False, encoding="utf-8")

    qa = make_qa(corpus, picked_a, picked_b)
    Path(args.qa).write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    print("✅ écrit:", args.output)
    print("✅ QA   :", args.qa)
    print("rows:", qa["rows_total"], "range:", qa["time_range"])
    print("by_channel:", qa["by_channel"])
    print("picked A:", picked_a)
    print("picked B:", picked_b)

if __name__ == "__main__":
    main()
