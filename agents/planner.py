import os
from typing import Dict, Any, List, Optional
from agents.llm_provider import get_llm
from utils.formatting import extract_json_from_text
from utils.logging import logger

PLANNER_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "planner.txt")

def load_prompt_template() -> str:
    """Loads prompt template from disk or returns default."""
    try:
        if os.path.exists(PLANNER_PROMPT_PATH):
            with open(PLANNER_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"Could not read planner prompt from file: {e}")
        
    return (
        "You are the Research Planner for Evidentia AI. "
        "Break down the research question into structured sub-tasks and 3-5 academic search queries. "
        "Return valid JSON with keys: research_question, tasks, search_queries, rationale."
    )

def plan_research(
    research_question: str,
    llm: Optional[Any] = None,
    demo_mode: bool = False
) -> Dict[str, Any]:
    """
    Agent 1 – Research Planner:
    Converts user research question into targeted tasks and search queries.
    """
    logger.info(f"Research Planner started for question: '{research_question}'")
    
    if llm is None:
        llm = get_llm(demo_mode=demo_mode)

    template = load_prompt_template()
    prompt = f"{template}\n\nQuestion: {research_question}"

    try:
        response = llm.invoke(prompt)
        text_output = response.content if hasattr(response, "content") else str(response)
        parsed = extract_json_from_text(text_output)
        
        if parsed and isinstance(parsed, dict) and "tasks" in parsed and "search_queries" in parsed:
            logger.info(f"Research Planner completed with {len(parsed.get('tasks', []))} tasks and {len(parsed.get('search_queries', []))} queries.")
            return {
                "research_question": research_question,
                "tasks": parsed.get("tasks", []),
                "search_queries": parsed.get("search_queries", []),
                "rationale": parsed.get("rationale", "Structured multi-angle literature decomposition.")
            }
    except Exception as e:
        logger.warning(f"Error during Planner LLM invocation: {e}. Utilizing fallback planner heuristics.")

    # Resilient fallback planner
    fallback_tasks = [
        f"Identify applications and current landscape of {research_question}",
        f"Retrieve recent peer-reviewed publications and experimental benchmarks",
        f"Identify commonly used models, architectures, and evaluation datasets",
        f"Compare quantitative findings, advantages, and limitations across papers",
        f"Identify contradictions and potential research gaps"
    ]
    fallback_queries = [
        f"{research_question} systematic review",
        f"{research_question} empirical benchmark evaluation",
        f"{research_question} limitations and future directions",
        f"{research_question} deep learning state of the art"
    ]

    logger.info("Planner completed using fallback heuristics.")
    return {
        "research_question": research_question,
        "tasks": fallback_tasks,
        "search_queries": fallback_queries,
        "rationale": "Heuristic research decomposition covering state-of-the-art, empirical findings, and limitations."
    }
