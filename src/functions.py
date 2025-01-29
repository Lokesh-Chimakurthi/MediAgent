from metapub import PubMedFetcher
import os
import requests


def fetch_articles(search_term):
    """
    Fetches articles from PubMed based on a search term.
    Args:
        search_term (str): The term to search for in PubMed.
    Returns:
        list: A list of dictionaries, each containing the following keys:
            - title (str): The title of the article.
            - abstract (str): The abstract of the article. If the abstract is
              not available, an empty string is returned.
            - authors (list): A list of authors of the article.
            - url (str): The URL of the article.
    """

    fetcher = PubMedFetcher(api_key=os.getenv("PUBMED"))
    pmids = fetcher.pmids_for_query(search_term, retmax=5)

    articles = []
    for pmid in pmids:
        article = fetcher.article_by_pmid(pmid)
        article_data = {
            "title": article.title,
            "abstract": article.abstract if article.abstract is not None else "",
            "authors": article.authors,
            "url": article.url,
        }
        articles.append(article_data)

    return articles


def fetch_clinical_trails(search_term):
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    headers = {"accept": "application/json"}
    params = {
        "query.term": "{search_term}",
        "filter.overallStatus": "COMPLETED",
        "sort": "@relevance",
        "pageSize": 20
    }
    response = requests.get(base_url, headers=headers, params=params)
    return response.json()


def get_clinical_trails(search_term):
    """
    Extracts relevant information from the clinicaltrials.gov API response
    for a medical question-answering agent.

    Args:
        output: The JSON response from the clinicaltrials.gov API.

    Returns:
        A list of dictionaries, where each dictionary contains extracted
        information from a single study relevant to answering a medical question.
    """

    extracted_data = []
    output = fetch_clinical_trails(search_term)
    for study in output.get("studies", []):
        if not study.get("hasResults"):
            continue

        protocol_section = study.get("protocolSection", {})

        # Extract relevant modules
        identification_module = protocol_section.get("identificationModule", {})
        status_module = protocol_section.get("statusModule", {})
        description_module = protocol_section.get("descriptionModule", {})
        design_module = protocol_section.get("designModule", {})
        outcomes_module = protocol_section.get("outcomesModule", {})

        # Extract specific data points
        nct_id = identification_module.get("nctId")
        brief_title = identification_module.get("briefTitle")
        last_update_post_date = status_module.get("lastUpdatePostDateStruct", {}).get("date")
        brief_summary = description_module.get("briefSummary")
        study_type = design_module.get("studyType")
        primary_outcomes = outcomes_module.get("primaryOutcomes", [])[-3:]

        # Store extracted information
        extracted_study_data = {
            "nct_id": nct_id,
            "title": brief_title,
            # "last_update_post_date": last_update_post_date,
            "abstract": brief_summary,
            # "study_type": study_type,
            "primary_outcomes": primary_outcomes
        }
        extracted_data.append(extracted_study_data)

    return extracted_data[:5]
