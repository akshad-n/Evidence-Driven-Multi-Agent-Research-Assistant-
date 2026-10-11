from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from graph.state import ResearchState
from agents.planner import plan_research
from agents.search import search_academic_sources
from agents.analyzer import analyze_papers
from agents.evidence import extract_evidence, generate_citation_map
from agents.critic import audit_research
from agents.report import generate_report
from utils.logging import logger

def planner_node(state: ResearchState) -> Dict[str, Any]:
    """Node 1: Formulates systematic research tasks and targeted queries."""
    logger.info("Executing Graph Node: Research Planner")
    try:
        plan = plan_research(
            research_question=state["research_question"],
            demo_mode=state.get("demo_mode", False)
        )
        return {
            "research_tasks": plan.get("tasks", []),
            "search_queries": plan.get("search_queries", [])
        }
    except Exception as e:
        logger.error(f"Planner node error: {e}")
        return {
            "research_tasks": [f"Study {state['research_question']}"],
            "search_queries": [state["research_question"]],
            "errors": state.get("errors", []) + [f"Planner node: {str(e)}"]
        }

def search_node(state: ResearchState) -> Dict[str, Any]:
    """Node 2: Dispatches multi-source queries across academic repositories."""
    logger.info("Executing Graph Node: Search Agent")
    try:
        current_iter = state.get("search_iteration", 0) + 1
        queries = state.get("search_queries", [state["research_question"]])
        
        # If this is a re-search iteration, append specialized queries
        if current_iter > 1:
            queries = [f"{q} empirical limitations" for q in queries]
            
        papers = search_academic_sources(
            queries=queries,
            max_papers=state.get("max_papers", 5),
            include_web=state.get("include_web", True),
            demo_mode=state.get("demo_mode", False),
            year_min=state.get("year_min"),
            year_max=state.get("year_max"),
            sources=state.get("sources")
        )
        return {
            "papers": papers,
            "search_iteration": current_iter
        }
    except Exception as e:
        logger.error(f"Search node error: {e}")
        return {
            "papers": state.get("papers", []),
            "errors": state.get("errors", []) + [f"Search node: {str(e)}"]
        }

def analyzer_node(state: ResearchState) -> Dict[str, Any]:
    """Node 3: Performs structured extraction across retrieved literature."""
    logger.info("Executing Graph Node: Paper Analysis Agent")
    try:
        analyses = analyze_papers(
            papers=state.get("papers", []),
            demo_mode=state.get("demo_mode", False)
        )
        return {"paper_analysis": analyses}
    except Exception as e:
        logger.error(f"Analyzer node error: {e}")
        return {
            "paper_analysis": [],
            "errors": state.get("errors", []) + [f"Analyzer node: {str(e)}"]
        }

def evidence_node(state: ResearchState) -> Dict[str, Any]:
    """Node 4: Connects claims with verbatim evidence and formal citations."""
    logger.info("Executing Graph Node: Evidence Agent")
    try:
        papers = state.get("papers", [])
        evidence_items = extract_evidence(
            research_question=state["research_question"],
            papers=papers,
            demo_mode=state.get("demo_mode", False)
        )
        citations = generate_citation_map(papers)
        return {
            "evidence": evidence_items,
            "citations": citations
        }
    except Exception as e:
        logger.error(f"Evidence node error: {e}")
        return {
            "evidence": [],
            "citations": {},
            "errors": state.get("errors", []) + [f"Evidence node: {str(e)}"]
        }

def critic_node(state: ResearchState) -> Dict[str, Any]:
    """Node 5: Audits evidence quality, finds contradictions, and isolates gaps."""
    logger.info("Executing Graph Node: Critic Agent")
    try:
        audit = audit_research(
            research_question=state["research_question"],
            evidence_items=state.get("evidence", []),
            paper_analyses=state.get("paper_analysis", []),
            demo_mode=state.get("demo_mode", False)
        )
        return {
            "critic_results": audit,
            "contradictions": audit.get("contradictions", []),
            "research_gaps": audit.get("research_gaps", [])
        }
    except Exception as e:
        logger.error(f"Critic node error: {e}")
        return {
            "critic_results": {"needs_more_research": False},
            "contradictions": [],
            "research_gaps": [],
            "errors": state.get("errors", []) + [f"Critic node: {str(e)}"]
        }

def critic_decision_router(state: ResearchState) -> Literal["search", "report"]:
    """Conditional Edge: Determines whether to re-search or proceed to report."""
    critic_res = state.get("critic_results", {})
    needs_more = critic_res.get("needs_more_research", False)
    iteration = state.get("search_iteration", 1)

    if needs_more and iteration < 2:
        logger.info("Critic requested additional research. Routing back to Search Agent.")
        return "search"
    
    logger.info("Evidence verified by Critic. Routing to Report Generator.")
    return "report"

def report_node(state: ResearchState) -> Dict[str, Any]:
    """Node 6: Produces final evidence-grounded research report."""
    logger.info("Executing Graph Node: Report Generator")
    try:
        report_md = generate_report(
            research_question=state["research_question"],
            papers=state.get("papers", []),
            paper_analyses=state.get("paper_analysis", []),
            evidence=state.get("evidence", []),
            contradictions=state.get("contradictions", []),
            research_gaps=state.get("research_gaps", []),
            demo_mode=state.get("demo_mode", False)
        )
        return {"final_report": report_md}
    except Exception as e:
        logger.error(f"Report node error: {e}")
        return {
            "final_report": f"# Report Generation Error\n\nFailed to generate report: {str(e)}",
            "errors": state.get("errors", []) + [f"Report node: {str(e)}"]
        }

def build_research_graph():
    """
    Constructs and compiles the complete LangGraph multi-agent workflow.
    """
    workflow = StateGraph(ResearchState)

    # Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("report", report_node)

    # Add Directed Edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "search")
    workflow.add_edge("search", "analyzer")
    workflow.add_edge("analyzer", "evidence")
    workflow.add_edge("evidence", "critic")

    # Add Conditional Edge from Critic
    workflow.add_conditional_edges(
        "critic",
        critic_decision_router,
        {
            "search": "search",
            "report": "report"
        }
    )

    workflow.add_edge("report", END)

    compiled_graph = workflow.compile()
    logger.info("LangGraph multi-agent research workflow successfully compiled.")
    return compiled_graph
