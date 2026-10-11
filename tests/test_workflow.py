import pytest
from graph.workflow import build_research_graph
from graph.state import ResearchState

def test_workflow_end_to_end():
    graph = build_research_graph()
    assert graph is not None

    initial_state: ResearchState = {
        "research_question": "What are the applications of Generative AI in healthcare?",
        "session_id": "integration-test-session",
        "demo_mode": True,
        "max_papers": 3,
        "include_web": False,
        "search_iteration": 0,
        "research_tasks": [],
        "search_queries": [],
        "papers": [],
        "paper_analysis": [],
        "evidence": [],
        "citations": {},
        "contradictions": [],
        "research_gaps": [],
        "critic_results": {},
        "final_report": "",
        "errors": []
    }

    final_state = graph.invoke(initial_state)

    # Assertions across each agent stage
    assert len(final_state["research_tasks"]) >= 3
    assert len(final_state["search_queries"]) >= 2
    assert len(final_state["papers"]) >= 1
    assert len(final_state["paper_analysis"]) >= 1
    assert len(final_state["evidence"]) >= 1
    assert len(final_state["citations"]) >= 1
    assert "audit_results" in final_state["critic_results"]
    assert len(final_state["final_report"]) > 200
    assert "## 1. Research Question" in final_state["final_report"]
    assert "## 13. References" in final_state["final_report"]
