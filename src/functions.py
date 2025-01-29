from metapub import PubMedFetcher
import os
import requests
import xml.etree.ElementTree as ET
import re


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
        "query.term": search_term,
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
            "title": brief_title,
            # "last_update_post_date": last_update_post_date,
            "abstract": brief_summary,
            # "study_type": study_type,
            'url': f"https://clinicaltrials.gov/study/{nct_id}",
            "primary_outcomes": primary_outcomes
        }
        extracted_data.append(extracted_study_data)

    return extracted_data[:5]


def _fetch_medline_plus_raw(search_term):
    """Fetches raw data from MedlinePlus API"""
    base_url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {
        "db": "healthTopics",
        "term": search_term,
        "retmax": "10",
        "rettype": "brief"
    }
    
    response = requests.get(base_url, params=params)
    return ET.fromstring(response.content)

def _clean_text(text):
    """Remove XML/HTML tags and clean up whitespace"""
    # Remove <span> tags
    text = re.sub(r'<span[^>]*>', '', text)
    text = text.replace('</span>', '')
    
    # Remove <p>, <ul>, <li> tags
    text = re.sub(r'</?p>', '', text)
    text = re.sub(r'</?ul>', '', text)
    text = re.sub(r'</?li>', '• ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_medline_plus(search_term):
    """
    Fetches and formats health topics from MedlinePlus based on a search term.
    
    Args:
        search_term (str): The term to search for in MedlinePlus.
    
    Returns:
        list: A list of dictionaries containing cleaned health topic information
    """
    root = _fetch_medline_plus_raw(search_term)
    
    results = []
    for doc in root.findall('.//document'):
        topic = {
            'title': '',
            'url': doc.get('url', ''),
            'summary': ''
        }
        
        for content in doc.findall('content'):
            name = content.get('name')
            text = ''.join(content.itertext())
            
            if name == 'title':
                topic['title'] = _clean_text(text)
            elif name == 'FullSummary':
                topic['summary'] = _clean_text(text)
        
        results.append(topic)
    
    return results[:5]