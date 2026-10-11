import os
import requests
from typing import List, Dict, Any
from utils.logging import logger

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Queries Semantic Scholar Graph API for academic papers.
    Handles API rate limits and key authentication gracefully.
    """
    papers: List[Dict[str, Any]] = []
    if not query or not query.strip():
        return papers

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"User-Agent": "Evidentia-AI-Research-Assistant/1.0"}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": min(max_results, 10),
        "fields": "title,authors,year,abstract,url,externalIds"
    }

    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=8
        )
        if response.status_code == 200:
            data = response.json()
            for item in data.get("data", []):
                authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
                ext_ids = item.get("externalIds") or {}
                doi = ext_ids.get("DOI") or "Not available"
                
                papers.append({
                    "title": item.get("title", "").strip(),
                    "authors": authors,
                    "year": item.get("year"),
                    "abstract": item.get("abstract") or "Not available",
                    "url": item.get("url") or (f"https://doi.org/{doi}" if doi != "Not available" else "Not available"),
                    "doi": doi,
                    "source": "semanticscholar"
                })
            logger.info(f"Semantic Scholar returned {len(papers)} papers for query '{query}'.")
        elif response.status_code == 429:
            logger.warning("Semantic Scholar is currently rate-limited. Continuing with available sources.")
        else:
            logger.warning(f"Semantic Scholar returned status {response.status_code}. Continuing.")
    except Exception as e:
        logger.warning(f"Semantic Scholar is currently unavailable ({e}). Continuing with available sources.")

    return papers
