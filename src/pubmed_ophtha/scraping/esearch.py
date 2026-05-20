"""Manage access to PubMed Esearch utils."""

from time import sleep
from typing import List, Optional

import requests

from pubmed_ophtha.const.urls import PUBMED_ESEARCH_URL


def search_pmc_with_keywords(
    keywords: List[str], maximum_results: Optional[int] = None, step_size: int = 100
) -> List[str]:
    """
    Return the PubMed Central ids of articles that match at least one of the keywords.

    Args:
        keywords (List[str]): List of keywords to match the articles to.
        maximum_results (Optional[int], optional): Maximum number of articles to \
            retrieve. Will retrieve all when set to None. Defaults to None.
        step_size (int, optional): Number of results to retrieve per call.

    Returns:
        List[str]: PubMed Central ids as strings

    """
    assert len(keywords) > 0, "Missing keywords!"
    return search_pmc(
        " OR ".join(keywords).strip(), maximum_results, step_size=step_size
    )


def search_pmc(
    search_term: str, maximum_results: Optional[int] = None, step_size: int = 100
) -> List[str]:
    """
    Return the PubMed Central ids of articles that match the search term.

    Args:
        search_term (str): Term that is used for searching. See https://www.ncbi.nlm.nih.gov/pmc/advanced/.
        maximum_results (Optional[int], optional): Maximum number of articles to \
            retrieve. Will retrieve all when set to None. Defaults to None.
        step_size (int, optional): Number of results to retrieve per call.

    Returns:
        List[str]: PubMed Central ids as strings

    """
    assert len(search_term) > 0, "Missing search term!"

    start_index = 0
    fetched_pmc_ids: List[str] = []

    while maximum_results is None or len(fetched_pmc_ids) < maximum_results:
        request_parameters = {
            "db": "pmc",
            "term": search_term,
            "retmax": step_size,
            "retmode": "json",
            "retstart": start_index,
        }

        response = requests.get(PUBMED_ESEARCH_URL, params=request_parameters)

        response_json = response.json()

        # cspell:disable-next-line
        fetched_pmc_ids.extend(response_json["esearchresult"]["idlist"])

        if maximum_results is None:
            # cspell:disable-next-line
            maximum_results = int(response_json["esearchresult"]["count"])

        start_index += step_size
        sleep(0.34)  # Maximum of 3 requests per second

    if len(fetched_pmc_ids) > maximum_results:
        fetched_pmc_ids = fetched_pmc_ids[:maximum_results]
    return fetched_pmc_ids
