import requests
from typing import List, Dict, Any
from utils.logging import logger

CROSSREF_API_URL = "https://api.crossref.org/works"

def search_crossref(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches Crossref API for registered DOI publications.
    """
    papers: List[Dict[str, Any]] = []
    if not query or not query.strip():
        return papers

    params = {
        "query": query,
        "rows": min(max_results, 10),
        "sort": "relevance",
        "select": "DOI,title,author,published-print,published-online,abstract,URL"
    }
    headers = {
        "User-Agent": "EvidentiaAI/1.0 (mailto:evidentia-research@example.com)"
    }

    try:
        response = requests.get(CROSSREF_API_URL, params=params, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            for item in items:
                title_list = item.get("title", [])
                title = title_list[0].strip() if title_list else "Untitled"
                
                authors = []
                for a in item.get("author", []):
                    name_parts = [a.get("given", ""), a.get("family", "")]
                    full_name = " ".join([p for p in name_parts if p]).strip()
                    if full_name:
                        authors.append(full_name)

                # Extract year
                year = None
                date_parts = (
                    item.get("published-print", {}).get("date-parts") or
                    item.get("published-online", {}).get("date-parts") or []
                )
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]

                abstract = item.get("abstract", "")
                if abstract and "<" in abstract:
                    # Strip basic XML tags like <jats:p>
                    import re
                    abstract = re.sub(r"<[^>]+>", "", abstract).strip()
                if not abstract:
                    abstract = "Not available"

                doi = item.get("DOI", "Not available")
                url = item.get("URL", f"https://doi.org/{doi}" if doi != "Not available" else "Not available")

                papers.append({
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "url": url,
                    "doi": doi,
                    "source": "crossref"
                })
            logger.info(f"Crossref returned {len(papers)} papers for '{query}'.")
        else:
            logger.warning(f"Crossref API returned status {response.status_code}. Continuing.")
    except Exception as e:
        logger.warning(f"Crossref API is currently unavailable ({e}). Continuing with other sources.")

    return papers
