#Copyright (c) 2024 Thomas Kientz
"""Configuration file for the ECB Scraper."""

ROOT_URL = "https://www.ecb.europa.eu"
BASE_INDEX_URL_1 = ROOT_URL + "/press/pressconf/{year}/html/index_include.en.html"
BASE_INDEX_URL_2 = ROOT_URL +"/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html"
MIN_YEAR = 1998
START_TAG = "Jump to the transcript of the questions and answers"
END_TAG = "\nReproduction is permitted provided that the source is acknowledged"
EXCLUDED_CLASSES = ["title", "address-box", "related-publications", "related-topics", "ecb-pressContentTitle"]


def index_url_year(year):
    """Get the URL for the index page of a given year."""
    if year >= 2020:
        return BASE_INDEX_URL_2.format(year=year)
    else:
        return BASE_INDEX_URL_1.format(year=year)

