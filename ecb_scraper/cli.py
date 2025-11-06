
"""Command line interface for the ECB Scraper."""

import argparse
import os
import pandas as pd
from .scraper import load_ecb_conferences
from .config import MIN_YEAR


def _write_double_slash_csv(
    df: pd.DataFrame,
    output_file: str,
    order: list[str] = ("date", "title", "link", "text"),
) -> None:
    """
    Write a custom `//`-separated text file with the exact header:
    date//title//link//text

    Rules:
      - Keep only the requested columns in that exact order.
      - Replace newlines in cell values with a single space so each record stays on one line.
      - Escape occurrences of `//` inside values to `\/\//` so they don't break parsing.
      - Encode as UTF-8 with BOM for better Excel compatibility.
    """
    lower_map = {c.lower(): c for c in df.columns}
    missing = [c for c in order if c.lower() not in lower_map]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    cols = [lower_map[c.lower()] for c in order]

    def sanitize(val) -> str:
        if pd.isna(val):
            return ""
        if isinstance(val, (pd.Timestamp, )):
            s = val.strftime("%Y-%m-%d")
        else:
            s = str(val)
        s = s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        s = s.replace("//", r"\/\//")
        return s

    header = "//".join(order)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        f.write(header + "\n")
        for _, row in df[cols].iterrows():
            f.write("//".join(sanitize(row[c]) for c in cols) + "\n")


def save_data(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the DataFrame in the specified format.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    output_file : str
        The path to the output file.
    """
    format_ = output_file.split(".")[-1].lower()

    if format_ == "csv":
        _write_double_slash_csv(df, output_file)
    elif format_ == "json":
        df.to_json(output_file, orient="records", force_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format_}")


def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(
        description="Fetch ECB press conferences and save them in a specified format."
    )
    parser.add_argument("--start-year", type=int, default=None, help="The start year for fetching conferences.")
    parser.add_argument("--end-year", type=int, default=None, help="The end year for fetching conferences.")
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="The path to the output file. Format must be CSV or JSON.",
    )

    args = parser.parse_args()

    if args.start_year is None:
        args.start_year = MIN_YEAR

    conferences_df = load_ecb_conferences(start_year=args.start_year, end_year=args.end_year)

    save_data(conferences_df, args.output_file)

    print(f"Data saved to {args.output_file}.")


if __name__ == "__main__":
    main()
