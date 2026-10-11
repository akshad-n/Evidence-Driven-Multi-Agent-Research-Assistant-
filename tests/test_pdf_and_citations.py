import pytest
from utils.pdf_generator import generate_research_pdf
from utils.formatting import generate_bibtex, parse_authors

def test_generate_research_pdf():
    papers = [
        {
            "title": "Clinical AI Evaluation Benchmark",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2024,
            "source": "nature",
            "doi": "10.1038/s41586-024-001",
            "url": "https://doi.org/10.1038/s41586-024-001"
        }
    ]
    analyses = [
        {
            "paper": "Clinical AI Evaluation Benchmark",
            "method": "Multi-center clinical trial",
            "dataset": "MIMIC-IV subset",
            "model": "ClinGen-7B",
            "results": "94.2% accuracy",
            "limitations": "Inference latency"
        }
    ]
    evidence = [
        {
            "claim": "Domain tuning significantly reduces clinical hallucination.",
            "evidence": "Observed 68% relative decrease in errors.",
            "source_title": "Clinical AI Evaluation Benchmark",
            "citation": "[1]",
            "confidence": 0.95
        }
    ]
    contradictions = [
        {
            "finding_a": "High accuracy in general pathology.",
            "source_a": "Paper A",
            "finding_b": "Severe failure modes in rare subspecialties.",
            "source_b": "Paper B",
            "explanation": "Divergence between common and long-tail conditions."
        }
    ]
    gaps = [
        {
            "gap": "Lack of prospective bedside clinical trials.",
            "reason": "Current benchmarks are retrospective.",
            "supporting_papers": ["Clinical AI Evaluation Benchmark"]
        }
    ]

    pdf_bytes = generate_research_pdf(
        research_question="What are clinical applications of GenAI?",
        papers=papers,
        paper_analyses=analyses,
        evidence=evidence,
        contradictions=contradictions,
        research_gaps=gaps,
        final_report_markdown="## 2. Executive Summary\nGenAI enhances EHR summarization.",
        session_id="test-session-123"
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # Must start with PDF magic bytes
    assert pdf_bytes.startswith(b"%PDF")

def test_generate_bibtex():
    papers = [
        {
            "title": "Large Language Models in Medicine",
            "authors": ["Karan Singhal", "Shekoofeh Azizi"],
            "year": 2023,
            "source": "nature",
            "doi": "10.1038/s41586-023-06291-2",
            "url": "https://doi.org/10.1038/s41586-023-06291-2"
        }
    ]
    bibtex = generate_bibtex(papers)
    assert "@article{" in bibtex
    assert "Singhal" in bibtex
    assert "2023" in bibtex
    assert "10.1038/s41586-023-06291-2" in bibtex

def test_parse_authors():
    assert parse_authors(["Author One", "Author Two"]) == ["Author One", "Author Two"]
    assert parse_authors("['John Doe', 'Jane Smith']") == ["John Doe", "Jane Smith"]
    assert parse_authors("Dr. Single Author") == ["Dr. Single Author"]
    assert parse_authors(None) == ["Unknown Authors"]
