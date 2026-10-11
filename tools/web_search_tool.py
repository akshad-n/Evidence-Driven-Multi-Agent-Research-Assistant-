import requests
import urllib.parse
from typing import List, Dict, Any
from utils.logging import logger

def search_trusted_web(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    Searches trusted academic web domains and public repositories.
    """
    papers: List[Dict[str, Any]] = []
    if not query or not query.strip():
        return papers

    encoded = urllib.parse.quote_plus(f"{query} site:nih.gov OR site:nature.com OR site:sciencedirect.com")
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(res.text, "html.parser")
            results = soup.find_all("div", class_="result", limit=max_results)
            for r in results:
                title_elem = r.find("a", class_="result__a")
                snippet_elem = r.find("a", class_="result__snippet")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get("href", "")
                    # Duckduckgo redirects /uddg?uddg=...
                    if "uddg=" in link:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                        link = parsed.get("uddg", [link])[0]
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else "Not available"
                    
                    papers.append({
                        "title": title,
                        "authors": ["Web Academic Source"],
                        "year": None,
                        "abstract": snippet,
                        "url": link,
                        "doi": "Not available",
                        "source": "web"
                    })
            logger.info(f"Trusted web search returned {len(papers)} results.")
    except Exception as e:
        logger.warning(f"Web search encountered an issue: {e}. Continuing.")

    return papers
