import json
import re
from typing import Any, Dict, List, Optional

def extract_json_from_text(text: str) -> Optional[Any]:
    """Extracts and parses JSON object or array from LLM text responses, handling markdown code blocks."""
    if not text:
        return None
        
    text = text.strip()
    
    # 1. Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # 2. Look for ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Look for curly braces or square brackets
    brace_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return None

def format_citation(paper: Dict[str, Any], index: int = 1) -> str:
    """Formats paper metadata into standard academic citation format."""
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        if len(authors) > 3:
            authors_str = f"{', '.join(authors[:3])}, et al."
        elif authors:
            authors_str = ", ".join(authors)
        else:
            authors_str = "Unknown Authors"
    else:
        authors_str = str(authors or "Unknown Authors")

    title = paper.get("title", "Untitled Paper").strip()
    year = paper.get("year", "n.d.")
    source = paper.get("source", "Academic Source").upper()
    url = paper.get("url", "")
    doi = paper.get("doi", "")

    citation = f"[{index}] {authors_str} ({year}). *{title}*. {source}."
    if doi:
        citation += f" DOI: {doi}."
    if url:
        citation += f" Available at: {url}"
    return citation

def parse_authors(authors_raw: Any) -> List[str]:
    """Safely parses author representation into list of strings without dangerous eval()."""
    if isinstance(authors_raw, list):
        return [str(a).strip() for a in authors_raw if str(a).strip()]
    if not authors_raw:
        return ["Unknown Authors"]
    
    s = str(authors_raw).strip()
    # Try JSON
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if str(a).strip()]
        except Exception:
            pass
        # Fallback regex split
        items = re.findall(r"['\"](.*?)['\"]", s)
        if items:
            return items

    return [s]

def generate_bibtex(papers: List[Dict[str, Any]]) -> str:
    """
    Generates standard BibTeX formatted entries for literature citation management
    compatible with Zotero, Mendeley, and LaTeX.
    """
    entries = []
    for idx, p in enumerate(papers):
        authors = parse_authors(p.get("authors", []))
        author_field = " and ".join(authors) if authors else "Unknown"
        title = p.get("title", f"Paper {idx + 1}").strip()
        year = str(p.get("year") or 2024)
        source = p.get("source", "misc").lower()
        doi = p.get("doi", "")
        url = p.get("url", "")
        
        # Create citation key: FirstAuthorLastName_Year_FirstWord
        first_author = authors[0].split()[-1] if authors else "Paper"
        first_author_clean = re.sub(r'[^a-zA-Z0-9]', '', first_author)
        first_title_word = re.sub(r'[^a-zA-Z0-9]', '', title.split()[0]) if title.split() else "ref"
        cite_key = f"{first_author_clean}{year}{first_title_word}"

        lines = [
            f"@article{{{cite_key},",
            f"  author    = {{{author_field}}},",
            f"  title     = {{{{{title}}}}},",
            f"  year      = {{{year}}},",
            f"  journal   = {{{source.upper()}}},"
        ]
        if doi and doi != "Not available":
            lines.append(f"  doi       = {{{doi}}},")
        if url and url != "Not available":
            lines.append(f"  url       = {{{url}}},")
        lines.append("}")
        entries.append("\n".join(lines))

    return "\n\n".join(entries)

