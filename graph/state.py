from typing import TypedDict, List, Dict, Any, Optional

class ResearchState(TypedDict):
    """
    Typed state representation for Evidentia AI LangGraph multi-agent workflow.
    """
    # Core research inputs
    research_question: str
    session_id: str
    demo_mode: bool
    max_papers: int
    include_web: bool
    search_iteration: int
    year_min: Optional[int]
    year_max: Optional[int]
    sources: Optional[List[str]]
    llm_provider: Optional[str]
    model_name: Optional[str]

    # Agent outputs
    research_tasks: List[str]
    search_queries: List[str]
    papers: List[Dict[str, Any]]
    paper_analysis: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    citations: Dict[str, Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    research_gaps: List[Dict[str, Any]]
    critic_results: Dict[str, Any]
    final_report: str
    errors: List[str]
