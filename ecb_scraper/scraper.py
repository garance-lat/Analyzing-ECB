##Copyright (c) 2024 Thomas Kientz
"""Main scraper and parser for ECB press conference transcripts."""

import pandas as pd
from tqdm import tqdm
from requests import Response, get
from bs4 import BeautifulSoup, element
from typing import List, Dict, Union, Optional
from ecb_scraper.config import index_url_year, MIN_YEAR, ROOT_URL, END_TAG, START_TAG, EXCLUDED_CLASSES


def get_year_conferences(year: int) -> pd.DataFrame:
    """
    Get all press conferences for a given year.

    Parameters
    ----------
    year : int
        The year for which to fetch the press conferences.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the date, title, link, and text of each conference.
    """
    index_url = index_url_year(year)

    response: Response = get(index_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    dt_elements = soup.find_all("dt")
    div_elements = soup.find_all("div", class_="title")

    output: List[Dict[str, Union[str, int]]] = []

    for dt, div in zip(dt_elements, div_elements):
        date = dt.text.strip()
        a_tag = div.find("a")
        if a_tag is None or not a_tag.get("href"):
            continue

        link = a_tag["href"]
        title = div.text.strip()
        output.append(
            {"date": date, "title": title, "link": ROOT_URL + link, "text": get_conference_text(ROOT_URL + link)}
        )

    return pd.DataFrame(output)


def get_conference_text(link: str) -> str:
    """
    Get the text of a press conference.

    Parameters
    ----------
    link : str
        The URL to the press conference.

    Returns
    -------
    str
        The text of the conference.
    """
    response: Response = get(link)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    main_tag = soup.find("main")

    # 🧠 Si la page n'a pas de <main>, on ignore (probablement une page vidéo/PDF)
    if main_tag is None:
        print(f"⚠️ Aucun tag <main> trouvé pour {link} — page ignorée.")
        return ""

    relevant_elements: List[str] = []

    for elem in main_tag.children:
        if not isinstance(elem, element.Tag):
            continue
        if not any(cls in elem.get("class", []) for cls in EXCLUDED_CLASSES):
            relevant_elements.append(elem.text.strip())

    text = "\n".join(relevant_elements)

    if START_TAG in text:
        return text.split(START_TAG)[-1]
    if END_TAG in text:
        return text.split(END_TAG)[0]

    return text


def load_ecb_conferences(start_year: int = MIN_YEAR, end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch all conferences from start_year up to end_year.

    Parameters
    ----------
    start_year : int, optional
        The year from which to start fetching conferences.
    end_year : int, optional
        The last year for which to fetch conferences. Defaults to the current year if None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all conferences from start_year to end_year.
    """
    if end_year is None:
        end_year = pd.Timestamp.now().year

    full_dataframe = pd.DataFrame()

    pbar = tqdm(range(start_year, end_year + 1), desc="Starting...")
    for year in pbar:
        year_df = get_year_conferences(year)
        full_dataframe = pd.concat([full_dataframe, year_df], ignore_index=True)

        pbar.set_description(f"Year: {year}/{end_year}, Total conferences: {len(full_dataframe)}")

    full_dataframe["date"] = pd.to_datetime(full_dataframe["date"])
    return full_dataframe.sort_values(by="date").reset_index(drop=True)
