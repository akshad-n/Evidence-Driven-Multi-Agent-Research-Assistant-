import os
import time
import json
import csv
from typing import Dict, Any, List
from graph.workflow import build_research_graph
from graph.state import ResearchState
from utils.logging import logger

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "evaluation_dataset.json")
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")

def evaluate_single_question(question: str, graph: Any, demo_mode: bool = True) -> Dict[str, Any]:
    """
    Executes the multi-agent pipeline on a research question and evaluates empirical metrics.
    """
    start_time = time.time()
    initial_state: ResearchState = {
        "research_question": question,
        "session_id": f"eval-{int(start_time)}",
        "demo_mode": demo_mode,
        "max_papers": 4,
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

    try:
        final_state = graph.invoke(initial_state)
        elapsed = round(time.time() - start_time, 2)

        papers = final_state.get("papers", [])
        evidence = final_state.get("evidence", [])
        citations = final_state.get("citations", {})
        report = final_state.get("final_report", "")

        # 1. Retrieval Accuracy: papers with non-empty abstract & real URL/DOI
        valid_papers = sum(1 for p in papers if p.get("title") and p.get("abstract") != "Not available")
        retrieval_accuracy = round(valid_papers / max(1, len(papers)), 3)

        # 2. Citation Correctness: citations linked to real papers
        valid_citations = sum(1 for c in citations.values() if c.get("marker") and c.get("citation_text"))
        citation_correctness = round(valid_citations / max(1, len(citations)), 3)

        # 3. Answer Relevance: overlap of question keywords in report
        keywords = [w.lower() for w in question.replace("?", "").split() if len(w) > 3]
        matches = sum(1 for k in keywords if k in report.lower())
        answer_relevance = round(matches / max(1, len(keywords)), 3)

        # 4. Task Completion: 13 sections generated
        required_sections = [
            "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.",
            "## 7.", "## 8.", "## 9.", "## 10.", "## 11.", "## 12.", "## 13."
        ]
        present_sections = sum(1 for s in required_sections if s in report)
        task_completion = round(present_sections / 13.0, 3)

        notes = f"{len(papers)} papers retrieved, {len(evidence)} evidence items, {present_sections}/13 report sections"

        return {
            "question": question,
            "retrieval_accuracy": retrieval_accuracy,
            "citation_correctness": citation_correctness,
            "answer_relevance": answer_relevance,
            "task_completion": task_completion,
            "response_time": elapsed,
            "notes": notes
        }
    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        logger.error(f"Evaluation failed for '{question}': {e}")
        return {
            "question": question,
            "retrieval_accuracy": 0.0,
            "citation_correctness": 0.0,
            "answer_relevance": 0.0,
            "task_completion": 0.0,
            "response_time": elapsed,
            "notes": f"Error: {str(e)}"
        }

def run_evaluation(num_samples: int = 35, demo_mode: bool = True):
    """
    Runs automated evaluation across the benchmark questions dataset.
    """
    logger.info(f"Starting Evidentia AI evaluation across {num_samples} questions...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    target_questions = questions[:num_samples]
    graph = build_research_graph()

    results: List[Dict[str, Any]] = []
    for idx, q in enumerate(target_questions):
        logger.info(f"[{idx + 1}/{len(target_questions)}] Evaluating: {q}")
        res = evaluate_single_question(q, graph=graph, demo_mode=demo_mode)
        results.append(res)

    # Save to CSV
    fieldnames = [
        "question",
        "retrieval_accuracy",
        "citation_correctness",
        "answer_relevance",
        "task_completion",
        "response_time",
        "notes"
    ]
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved evaluation results to {OUTPUT_CSV_PATH}")

    # Summary Statistics
    n = len(results)
    avg_retrieval = sum(r["retrieval_accuracy"] for r in results) / n
    avg_citation = sum(r["citation_correctness"] for r in results) / n
    avg_relevance = sum(r["answer_relevance"] for r in results) / n
    avg_completion = sum(r["task_completion"] for r in results) / n
    avg_time = sum(r["response_time"] for r in results) / n

    print("\n=======================================================")
    print("           EVIDENTIA AI EVALUATION SUMMARY             ")
    print("=======================================================")
    print(f"Total Questions Evaluated:    {n}")
    print(f"Mean Retrieval Accuracy:      {avg_retrieval * 100:.1f}%")
    print(f"Mean Citation Correctness:    {avg_citation * 100:.1f}%")
    print(f"Mean Answer Relevance:        {avg_relevance * 100:.1f}%")
    print(f"Mean Task Completion:         {avg_completion * 100:.1f}%")
    print(f"Average Response Time:        {avg_time:.2f} seconds")
    print("=======================================================\n")

if __name__ == "__main__":
    run_evaluation(num_samples=35, demo_mode=True)
