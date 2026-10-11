import pytest
from agents.planner import plan_research
from agents.search import search_academic_sources, deduplicate_papers, normalize_title
from tools.arxiv_tool import search_arxiv
from tools.crossref_tool import search_crossref

def test_planner_output_structure():
    question = "What are the applications of Generative AI in healthcare?"
    plan = plan_research(question, demo_mode=True)
    
    assert "research_question" in plan
    assert plan["research_question"] == question
    assert "tasks" in plan
    assert len(plan["tasks"]) >= 3
    assert "search_queries" in plan
    assert len(plan["search_queries"]) >= 2
    assert "rationale" in plan

def test_deduplication():
    papers = [
        {"title": "Paper One: GenAI in Health", "doi": "10.1234/a", "source": "arxiv"},
        {"title": "Paper One: GenAI in Health", "doi": "10.1234/a", "source": "crossref"},
        {"title": "Paper Two: Transformers", "doi": "10.1234/b", "source": "pubmed"},
    ]
    unique = deduplicate_papers(papers)
    assert len(unique) == 2
    assert unique[0]["title"] == "Paper One: GenAI in Health"
    assert unique[1]["title"] == "Paper Two: Transformers"

def test_search_agent_demo_mode():
    queries = ["generative AI healthcare", "clinical transformer models"]
    papers = search_academic_sources(queries, max_papers=3, demo_mode=True)
    
    assert len(papers) == 3
    for p in papers:
        assert "title" in p and p["title"]
        assert "authors" in p and isinstance(p["authors"], list)
        assert "abstract" in p and p["abstract"]
        assert "url" in p and p["url"]
        assert "doi" in p
        assert "source" in p

def test_search_tools_empty_query():
    assert search_arxiv("") == []
    assert search_crossref("") == []
