import pytest
from agents.analyzer import analyze_single_paper, analyze_papers
from agents.evidence import extract_evidence, generate_citation_map
from data.sample_papers import get_demo_papers

def test_paper_analysis_agent():
    sample_paper = {
        "title": "A Survey of Generative AI in Healthcare",
        "authors": ["Chun-Kai Wang", "Chun-Yen Chen"],
        "year": 2024,
        "abstract": "We synthesize 120 studies showing clinical documentation time reductions of up to 40%."
    }
    analysis = analyze_single_paper(sample_paper, demo_mode=True)
    
    assert analysis["paper"] == sample_paper["title"]
    assert "method" in analysis
    assert "dataset" in analysis
    assert "model" in analysis
    assert "results" in analysis
    assert "limitations" in analysis
    assert "future_work" in analysis
    assert isinstance(analysis["key_findings"], list)

def test_evidence_agent():
    papers = get_demo_papers(limit=2)
    question = "What are the applications of Generative AI in healthcare?"
    evidence = extract_evidence(question, papers, demo_mode=True)
    
    assert len(evidence) >= 1
    for item in evidence:
        assert "claim" in item and len(item["claim"]) > 5
        assert "evidence" in item and len(item["evidence"]) > 5
        assert "source_title" in item
        assert "source_url" in item
        assert "doi" in item
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0
        assert "citation" in item

def test_citation_mapping():
    papers = get_demo_papers(limit=3)
    cmap = generate_citation_map(papers)
    assert len(cmap) == 3
    for title, cinfo in cmap.items():
        assert cinfo["marker"].startswith("[") and cinfo["marker"].endswith("]")
        assert "citation_text" in cinfo
