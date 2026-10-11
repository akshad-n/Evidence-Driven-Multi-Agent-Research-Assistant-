import os
import json
from typing import Dict, Any, List, Optional
from agents.llm_provider import get_llm
from utils.formatting import extract_json_from_text
from utils.logging import logger

ANALYZER_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "analyzer.txt")

def load_analyzer_prompt() -> str:
    """Loads prompt template from disk or returns default."""
    try:
        if os.path.exists(ANALYZER_PROMPT_PATH):
            with open(ANALYZER_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"Could not read analyzer prompt: {e}")
        
    return (
        "Analyze the following research paper and extract: method, dataset, model, results, "
        "advantages, limitations, future_work, key_findings. Use 'Not specified in source' "
        "if information is not in the text."
    )

def analyze_single_paper(
    paper: Dict[str, Any],
    llm: Optional[Any] = None,
    demo_mode: bool = False
) -> Dict[str, Any]:
    """
    Analyzes an individual paper to extract structured methodological and empirical details.
    Enforces anti-hallucination standards.
    """
    title = paper.get("title", "Untitled Paper")
    authors = ", ".join(paper.get("authors", [])) or "Not specified"
    year = paper.get("year", "Not specified")
    content = paper.get("abstract", "") or paper.get("content", "")

    if llm is None:
        llm = get_llm(demo_mode=demo_mode)

    template = load_analyzer_prompt()
    prompt = (
        template.replace("{title}", title)
        .replace("{authors}", authors)
        .replace("{year}", str(year))
        .replace("{content}", content)
    )

    try:
        response = llm.invoke(prompt)
        text_output = response.content if hasattr(response, "content") else str(response)
        parsed = extract_json_from_text(text_output)
        
        if parsed and isinstance(parsed, dict):
            # Ensure all required keys exist and default missing to 'Not specified in source'
            return {
                "paper": title,
                "method": parsed.get("method") or "Not specified in source",
                "dataset": parsed.get("dataset") or "Not specified in source",
                "model": parsed.get("model") or "Not specified in source",
                "results": parsed.get("results") or "Not specified in source",
                "advantages": parsed.get("advantages") or "Not specified in source",
                "limitations": parsed.get("limitations") or "Not specified in source",
                "future_work": parsed.get("future_work") or "Not specified in source",
                "key_findings": parsed.get("key_findings", [])
            }
    except Exception as e:
        logger.warning(f"LLM paper analysis encountered an issue for '{title}': {e}")

    # Deterministic extraction fallback strictly adhering to anti-hallucination rules
    key_findings = []
    if content and content != "Not available":
        # Extract first 2 sentences as findings
        sentences = [s.strip() for s in content.split(".") if len(s.strip()) > 20]
        key_findings = sentences[:2]
    else:
        key_findings = ["Abstract or full text not provided in retrieval metadata."]

    return {
        "paper": title,
        "method": "Not specified in source",
        "dataset": "Not specified in source",
        "model": "Not specified in source",
        "results": "Not specified in source",
        "advantages": "Not specified in source",
        "limitations": "Not specified in source",
        "future_work": "Not specified in source",
        "key_findings": key_findings
    }

def analyze_papers(
    papers: List[Dict[str, Any]],
    llm: Optional[Any] = None,
    demo_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Agent 3 – Paper Analysis Agent:
    Iterates over collected papers and generates structured empirical breakdowns.
    """
    logger.info(f"Paper Analysis Agent started for {len(papers)} papers.")
    analyses: List[Dict[str, Any]] = []

    for p in papers:
        analysis = analyze_single_paper(p, llm=llm, demo_mode=demo_mode)
        analyses.append(analysis)

    logger.info(f"Paper Analysis Agent completed {len(analyses)} analyses.")
    return analyses
