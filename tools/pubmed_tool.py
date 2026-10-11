import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from utils.logging import logger

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

def search_pubmed(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches PubMed via NCBI E-utilities API.
    """
    papers: List[Dict[str, Any]] = []
    if not query or not query.strip():
        return papers

    api_key = os.getenv("NCBI_API_KEY", "").strip()
    base_params = {"email": "evidentia-ai@example.com", "tool": "EvidentiaAI"}
    if api_key:
        base_params["api_key"] = api_key

    try:
        # Step 1: E-Search for PMIDs
        search_params = {
            **base_params,
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 8),
            "retmode": "json",
            "sort": "relevance"
        }
        res = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=8)
        if res.status_code != 200:
            return papers
            
        id_list = res.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return papers

        # Step 2: E-Summary for metadata
        summary_params = {
            **base_params,
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        sum_res = requests.get(PUBMED_SUMMARY_URL, params=summary_params, timeout=8)
        if sum_res.status_code != 200:
            return papers

        result_dict = sum_res.json().get("result", {})
        for pmid in id_list:
            item = result_dict.get(pmid)
            if not item:
                continue

            title = item.get("title", "").strip().rstrip(".")
            authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
            pubdate = item.get("pubdate", "")
            year = None
            if pubdate:
                # Year is typically first 4 digits
                import re
                m = re.search(r"\b(19\d\d|20\d\d)\b", pubdate)
                if m:
                    year = int(m.group(1))

            doi = "Not available"
            for article_id in item.get("articleids", []):
                if article_id.get("idtype") == "doi":
                    doi = article_id.get("value")
                    break

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            papers.append({
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": f"PubMed publication PMID:{pmid}. Abstract indexed in MEDLINE database.",
                "url": url,
                "doi": doi,
                "source": "pubmed"
            })

        logger.info(f"PubMed returned {len(papers)} papers for query '{query}'.")
    except Exception as e:
        logger.warning(f"PubMed search is currently unavailable ({e}). Continuing with other sources.")

    return papers
