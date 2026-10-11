import re
from typing import List, Dict, Any, Optional
from tools.arxiv_tool import search_arxiv
from tools.semantic_scholar_tool import search_semantic_scholar
from tools.crossref_tool import search_crossref
from tools.pubmed_tool import search_pubmed
from tools.web_search_tool import search_trusted_web
from data.sample_papers import get_demo_papers
from utils.logging import logger

def normalize_title(title: str) -> str:
    """Normalizes paper title for reliable deduplication."""
    return re.sub(r'[^a-zA-Z0-9]', '', title.lower())

def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates papers based on normalized title or valid DOI."""
    unique: List[Dict[str, Any]] = []
    seen_titles = set()
    seen_dois = set()

    for paper in papers:
        title = paper.get("title", "").strip()
        if not title:
            continue
            
        norm = normalize_title(title)
        doi = paper.get("doi", "").strip()

        if norm in seen_titles:
            continue
        if doi and doi != "Not available" and doi in seen_dois:
            continue

        seen_titles.add(norm)
        if doi and doi != "Not available":
            seen_dois.add(doi)

        unique.append(paper)

    return unique

def search_academic_sources(
    queries: List[str],
    max_papers: int = 5,
    include_web: bool = True,
    demo_mode: bool = False,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    sources: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Agent 2 – Search Agent:
    Dispatches targeted queries across academic repositories (arXiv, Semantic Scholar,
    Crossref, PubMed, Trusted Web) with automated deduplication, source selection, and year filtering.
    """
    logger.info(f"Search started with {len(queries)} queries (Target: {max_papers} papers).")
    collected_papers: List[Dict[str, Any]] = []

    # Normalized enabled sources
    active_sources = [s.lower().strip() for s in sources] if sources else ["arxiv", "semantic_scholar", "crossref", "pubmed", "web"]

    if demo_mode:
        logger.info("Demo Mode active: Loading verified benchmark papers.")
        primary_query = queries[0] if queries else ""
        papers = get_demo_papers(primary_query, limit=max_papers)
        if year_min or year_max:
            filtered = []
            for p in papers:
                y = p.get("year")
                if y:
                    if year_min and y < year_min:
                        continue
                    if year_max and y > year_max:
                        continue
                filtered.append(p)
            return filtered or papers[:max_papers]
        return papers

    # Dispatch queries across external repositories
    for q in queries[:3]:  # Top 3 queries to avoid excessive latency
        # 1. arXiv
        if "arxiv" in active_sources:
            try:
                arxiv_results = search_arxiv(q, max_results=3)
                collected_papers.extend(arxiv_results)
            except Exception as e:
                logger.warning(f"arXiv search error: {e}")

        # 2. Semantic Scholar
        if "semantic_scholar" in active_sources:
            try:
                ss_results = search_semantic_scholar(q, max_results=3)
                collected_papers.extend(ss_results)
            except Exception as e:
                logger.warning(f"Semantic Scholar search error: {e}")

        # 3. Crossref
        if "crossref" in active_sources:
            try:
                crossref_results = search_crossref(q, max_results=3)
                collected_papers.extend(crossref_results)
            except Exception as e:
                logger.warning(f"Crossref search error: {e}")

        # 4. PubMed
        if "pubmed" in active_sources:
            try:
                pubmed_results = search_pubmed(q, max_results=2)
                collected_papers.extend(pubmed_results)
            except Exception as e:
                logger.warning(f"PubMed search error: {e}")

        # 5. Trusted Web (optional)
        if include_web and "web" in active_sources:
            try:
                web_results = search_trusted_web(q, max_results=2)
                collected_papers.extend(web_results)
            except Exception as e:
                logger.warning(f"Web search error: {e}")

    # Deduplicate results
    unique_papers = deduplicate_papers(collected_papers)
    logger.info(f"Deduplication: {len(collected_papers)} raw results -> {len(unique_papers)} unique papers.")

    # Filter by publication year range if requested
    if year_min or year_max:
        year_filtered = []
        for p in unique_papers:
            y = p.get("year")
            if y:
                if year_min and y < year_min:
                    continue
                if year_max and y > year_max:
                    continue
            year_filtered.append(p)
        if year_filtered:
            unique_papers = year_filtered
            logger.info(f"Filtered by publication years ({year_min}-{year_max}): {len(unique_papers)} papers remaining.")

    # If external APIs returned fewer than minimum required papers (e.g. offline/rate-limited),
    # augment with verified demo papers to ensure resilience.
    if len(unique_papers) < 2:
        logger.info("External searches returned insufficient papers. Augmenting with verified academic benchmark corpus.")
        demo_papers = get_demo_papers(queries[0] if queries else "", limit=max_papers)
        unique_papers = deduplicate_papers(unique_papers + demo_papers)

    # Rank and trim to requested max_papers
    final_papers = unique_papers[:max_papers]
    logger.info(f"Search Agent finished: Retrieved {len(final_papers)} verified papers.")
    return final_papers
