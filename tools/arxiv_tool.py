import arxiv
from typing import List, Dict, Any
from utils.logging import logger

def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches arXiv for academic papers matching the query.
    Returns standardized paper dictionaries.
    """
    papers: List[Dict[str, Any]] = []
    if not query or not query.strip():
        return papers

    try:
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=1.0,
            num_retries=2
        )
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for result in client.results(search):
            year = result.published.year if result.published else None
            authors = [author.name for author in result.authors] if result.authors else []
            
            papers.append({
                "title": result.title.replace("\n", " ").strip(),
                "authors": authors,
                "year": year,
                "abstract": result.summary.replace("\n", " ").strip() if result.summary else "Not available",
                "url": result.entry_id,
                "doi": result.doi or f"10.48550/arXiv.{result.get_short_id()}",
                "source": "arxiv"
            })
            
        logger.info(f"arXiv search for '{query}' returned {len(papers)} papers.")
    except Exception as e:
        logger.warning(f"arXiv search encountered an issue: {e}. Continuing with other sources.")

    return papers
