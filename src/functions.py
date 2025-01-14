from metapub import PubMedFetcher
import os


def fetch_articles(search_term):
    fetcher = PubMedFetcher(api_key=os.getenv("PUBMED"))
    pmids = fetcher.pmids_for_query(search_term, retmax=5)

    articles = []
    for pmid in pmids:
        article = fetcher.article_by_pmid(pmid)
        article_data = {
            "title": article.title,
            "abstract": article.abstract,
            "authors": article.authors,
            "url": article.url,
        }
        articles.append(article_data)

    return articles
