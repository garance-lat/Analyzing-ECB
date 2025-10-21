#Copyright (c) 2024 Thomas Kient
"""Command line interface for the ECB Scraper."""

import argparse
import pandas as pd
from .scraper import load_ecb_conferences
from .config import MIN_YEAR


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
    format_ = output_file.split(".")[-1]

    if format_.lower() == "csv":
        df.to_csv(output_file, index=False)
    elif format_.lower() == "json":
        df.to_json(output_file, orient="records")
    else:
        raise ValueError(f"Unsupported format: {format_}")


def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Fetch ECB press conferences and save them in a specified format.")
    parser.add_argument("--start-year", type=int, default=None, help="The start year for fetching conferences.")
    parser.add_argument("--end-year", type=int, default=None, help="The end year for fetching conferences.")
    parser.add_argument(
        "--output-file", type=str, required=True, help="The path to the output file. Format must be CSV or JSON."
    )

    args = parser.parse_args()

    if args.start_year is None:
        args.start_year = MIN_YEAR  # Make sure MIN_YEAR is imported or defined somewhere

    conferences_df = load_ecb_conferences(start_year=args.start_year, end_year=args.end_year)

    save_data(conferences_df, args.output_file)

    print(f"Data saved to {args.output_file} format.")


if __name__ == "__main__":
    main()
