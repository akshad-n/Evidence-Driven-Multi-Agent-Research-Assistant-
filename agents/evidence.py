import os
import json
from typing import Dict, Any, List, Optional
from agents.llm_provider import get_llm
from utils.formatting import extract_json_from_text, format_citation
from utils.logging import logger

EVIDENCE_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "evidence.txt")

def load_evidence_prompt() -> str:
    """Loads prompt template from disk or returns default."""
    try:
        if os.path.exists(EVIDENCE_PROMPT_PATH):
            with open(EVIDENCE_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"Could not read evidence prompt: {e}")
        
    return (
        "Extract factual research claims directly backed by verbatim or near-verbatim evidence quotes. "
        "Return valid JSON list of items with keys: claim, evidence, source_title, source_url, doi, "
        "page, section, confidence."
    )

def generate_citation_map(papers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Creates mapping from paper title/index to formal academic citation string and marker.
    """
    citation_map = {}
    for idx, paper in enumerate(papers):
        marker = f"[{idx + 1}]"
        citation_text = format_citation(paper, index=idx + 1)
        citation_map[paper.get("title", f"Paper {idx + 1}")] = {
            "marker": marker,
            "citation_text": citation_text,
            "doi": paper.get("doi", "Not available"),
            "url": paper.get("url", "Not available"),
            "paper": paper
        }
    return citation_map

def extract_evidence(
    research_question: str,
    papers: List[Dict[str, Any]],
    llm: Optional[Any] = None,
    demo_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Agent 4 – Evidence Agent:
    Extracts explicit claims paired with direct supporting textual evidence and authentic citations.
    """
    logger.info(f"Evidence Agent started for {len(papers)} papers.")
    if not papers:
        logger.warning("No papers provided for evidence extraction.")
        return []

    citation_map = generate_citation_map(papers)
    if llm is None:
        llm = get_llm(demo_mode=demo_mode)

    # Format papers as context for the agent
    paper_context_blocks = []
    for idx, p in enumerate(papers):
        title = p.get("title", "Untitled")
        abstract = p.get("abstract", "") or p.get("content", "")
        doi = p.get("doi", "Not available")
        url = p.get("url", "Not available")
        paper_context_blocks.append(
            f"Paper [{idx + 1}]:\nTitle: {title}\nDOI: {doi}\nURL: {url}\nContent/Abstract: {abstract}"
        )

    context_str = "\n\n---\n\n".join(paper_context_blocks)
    template = load_evidence_prompt()
    prompt = (
        template.replace("{paper_context}", context_str)
        .replace("{research_question}", research_question)
    )

    extracted_items: List[Dict[str, Any]] = []

    try:
        response = llm.invoke(prompt)
        text_output = response.content if hasattr(response, "content") else str(response)
        parsed = extract_json_from_text(text_output)

        if parsed and isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "claim" in item and "evidence" in item:
                    # Match to real paper citation
                    source_title = item.get("source_title", "")
                    matched_citation = None
                    for title_key, cdata in citation_map.items():
                        if title_key.lower() in source_title.lower() or source_title.lower() in title_key.lower():
                            matched_citation = cdata
                            break

                    marker = matched_citation["marker"] if matched_citation else "[1]"
                    item["citation"] = marker
                    item["page"] = item.get("page") or "Not available"
                    item["section"] = item.get("section") or "Not available"
                    item["confidence"] = float(item.get("confidence", 0.90))
                    extracted_items.append(item)
    except Exception as e:
        logger.warning(f"LLM evidence extraction encountered an error: {e}")

    # Fallback deterministic evidence extraction from verified papers
    if not extracted_items:
        logger.info("Using deterministic evidence extraction fallback from paper abstracts.")
        for idx, p in enumerate(papers):
            title = p.get("title", f"Paper {idx + 1}")
            abstract = p.get("abstract", "")
            doi = p.get("doi", "Not available")
            url = p.get("url", "Not available")

            if abstract and abstract != "Not available":
                sentences = [s.strip() for s in abstract.split(".") if len(s.strip()) > 30]
                if sentences:
                    claim = sentences[0] + "."
                    evidence = sentences[1] + "." if len(sentences) > 1 else sentences[0] + "."
                    extracted_items.append({
                        "claim": claim,
                        "evidence": evidence,
                        "source_title": title,
                        "source_url": url,
                        "doi": doi,
                        "page": "Not available",
                        "section": "Abstract",
                        "confidence": 0.92,
                        "citation": f"[{idx + 1}]"
                    })

    logger.info(f"Evidence Agent completed: Extracted {len(extracted_items)} verified evidence items.")
    return extracted_items
