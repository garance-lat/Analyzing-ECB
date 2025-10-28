from __future__ import annotations
import pandas as pd
from pathlib import Path

REQUIRED_COLS = ["date","title","link","text"]

def normalize_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    # normalize column names
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in source '{source_label}'")
    out = df[REQUIRED_COLS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.date
    out["source"] = source_label
    out = out.dropna(subset=["date","text"]).drop_duplicates(subset=["date","title","source"])
    return out

def merge_sources(speeches_csv: Path, pressers_csv: Path) -> pd.DataFrame:
    sp = pd.read_csv(speeches_csv)
    pr = pd.read_csv(pressers_csv)
    sp_norm = normalize_df(sp, "speech")
    pr_norm = normalize_df(pr, "presser")
    merged = pd.concat([sp_norm, pr_norm], ignore_index=True).sort_values("date")
    merged["doc_id"] = merged.index.astype(str)
    return merged

def save_merged(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
