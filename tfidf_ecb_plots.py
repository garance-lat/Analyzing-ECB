
import argparse
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def sanitize_filename(name: str, maxlen: int = 60) -> str:
    # Make a safe filename from an arbitrary string.
    if name is None or str(name).strip() == "":
        return "NA"
    s = re.sub(r"[^\w\-]+", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:maxlen] or "NA")

def ensure_outdir(csv_path: Path, outdir: str | None) -> Path:
    if outdir:
        out_path = Path(outdir)
    else:
        out_path = csv_path.parent / "outputs"
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path

def load_df(csv_path: Path, sep: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
    # Standardize column names
    df.columns = df.columns.str.strip()
    return df

def build_vectorizer(stop_words: str, max_df: float, min_df: int, ngram: tuple[int, int], max_features: int | None):
    # stop_words can be 'english', 'french', or 'none'
    sw = None if (stop_words is None or stop_words.lower() == "none") else stop_words.lower()
    vec = TfidfVectorizer(
        input="content",
        stop_words=sw,
        strip_accents="unicode",
        lowercase=True,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram,
        max_features=max_features
    )
    return vec

def compute_tfidf(df: pd.DataFrame, text_col: str, vec: TfidfVectorizer):
    texts = df[text_col].fillna("").astype(str).str.strip()
    X = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()
    return X, feature_names

def plot_barh(labels, values, title: str, save_path: Path):
    # One chart per figure; do not set any specific colors or styles.
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels)*0.4)))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.set_xlabel("TF-IDF")
    ax.set_title(title)
    # Show largest at top
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def top_terms_global(X, feature_names, top_n: int):
    avg = np.asarray(X.mean(axis=0)).ravel()
    idx = avg.argsort()[::-1][:top_n]
    terms = [feature_names[i] for i in idx]
    vals = [float(avg[i]) for i in idx]
    return terms, vals

def group_concat(df: pd.DataFrame, group_col: str, text_col: str) -> pd.DataFrame:
    # Concatenate texts per group
    grp = df.groupby(group_col, dropna=False)[text_col].apply(lambda s: " \n ".join(s.astype(str))).reset_index()
    # Replace NaN group names
    grp[group_col] = grp[group_col].fillna("NA")
    return grp

def top_terms_for_row(vec: TfidfVectorizer, feature_names, X_row_sparse, top_n: int):
    row = X_row_sparse.toarray().ravel()
    idx = row.argsort()[::-1]
    # Filter strictly positive weights
    idx = [i for i in idx if row[i] > 0][:top_n]
    terms = [feature_names[i] for i in idx]
    vals = [float(row[i]) for i in idx]
    return terms, vals

def main():
    parser = argparse.ArgumentParser(description="TF-IDF plots for ECB speeches (top terms global + per speaker).")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--sep", default="|", help="CSV separator (default: '|')")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig)")
    parser.add_argument("--text-col", default="contents", help="Name of text column (default: contents)")
    parser.add_argument("--group-col", default="speakers", help="Group column for per-speaker plots (default: speakers)")
    parser.add_argument("--stop-words", default="english", choices=["english","french","none"], help="Stop words (default: english)")
    parser.add_argument("--max-df", type=float, default=0.85, help="Ignore terms in >max_df fraction of docs (default: 0.85)")
    parser.add_argument("--min-df", type=int, default=3, help="Ignore terms that appear in <min_df docs (default: 3)")
    parser.add_argument("--ngram-min", type=int, default=1, help="Min n-gram (default: 1)")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max n-gram (default: 2)")
    parser.add_argument("--max-features", type=int, default=None, help="Cap vocabulary size (default: None)")
    parser.add_argument("--top-n", type=int, default=20, help="How many top terms to show (default: 20)")
    parser.add_argument("--top-speakers", type=int, default=5, help="How many speakers to plot (default: 5)")
    parser.add_argument("--outdir", default=None, help="Where to save outputs (default: CSV folder /outputs)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = ensure_outdir(csv_path, args.outdir)

    print(f"Loading: {csv_path}")
    df = load_df(csv_path, sep=args.sep, encoding=args.encoding)

    if args.text_col not in df.columns:
        raise SystemExit(f"Text column '{args.text_col}' not found. Available: {list(df.columns)}")

    # Build and fit vectorizer on the entire corpus
    vec = build_vectorizer(
        stop_words=args.stop_words,
        max_df=args.max_df,
        min_df=args.min_df,
        ngram=(args.ngram_min, args.ngram_max),
        max_features=args.max_features
    )
    X, feature_names = compute_tfidf(df, args.text_col, vec)

    # ---- Global top terms ----
    terms, vals = top_terms_global(X, feature_names, top_n=args.top_n)
    global_png = outdir / "top_terms_global.png"
    global_csv = outdir / "top_terms_global.csv"
    plot_barh(terms, vals, f"Top {args.top_n} global TF-IDF terms", global_png)
    pd.DataFrame({"term": terms, "avg_tfidf": vals}).to_csv(global_csv, index=False)
    print(f"Saved global top terms: {global_png} / {global_csv}")

    # ---- Per-speaker top terms ----
    group_col = args.group_col
    if group_col not in df.columns:
        print(f"[WARN] Group column '{group_col}' not found. Skipping per-group plots.")
        return

    # Choose top N speakers by document count
    top_speakers = (
        df[group_col].fillna("NA").value_counts().head(args.top_speakers).index.tolist()
    )
    print(f"Top {args.top_speakers} groups by doc count: {top_speakers}")

    # Transform concatenated texts for those speakers using the same fitted vectorizer
    # (This keeps the vocabulary consistent across plots)
    grp = group_concat(df, group_col, args.text_col)
    grp = grp[grp[group_col].isin(top_speakers)].reset_index(drop=True)

    grp_texts = grp[args.text_col].fillna("").astype(str).tolist()
    X_grp = vec.transform(grp_texts)

    for i, row_name in enumerate(grp[group_col].tolist()):
        terms_i, vals_i = top_terms_for_row(vec, feature_names, X_grp.getrow(i), top_n=args.top_n)
        safe = sanitize_filename(str(row_name))
        png_path = outdir / f"top_terms_{group_col}_{safe}.png"
        csv_path_i = outdir / f"top_terms_{group_col}_{safe}.csv"
        plot_barh(terms_i, vals_i, f"Top {args.top_n} terms — {group_col} = {row_name}", png_path)
        pd.DataFrame({"term": terms_i, "tfidf": vals_i}).to_csv(csv_path_i, index=False)
        print(f"Saved per-group top terms for '{row_name}': {png_path} / {csv_path_i}")

    # Optionally, also save the fitted vectorizer and a sparse matrix for reuse
    try:
        from scipy.sparse import save_npz
        import joblib
        save_npz(outdir / "tfidf_matrix.npz", X)
        joblib.dump(vec, outdir / "tfidf_vectorizer.joblib")
        print("Saved tfidf_matrix.npz and tfidf_vectorizer.joblib")
    except Exception as e:
        print(f"[INFO] Could not save matrix/vectorizer: {e}")

if __name__ == "__main__":
    main()
