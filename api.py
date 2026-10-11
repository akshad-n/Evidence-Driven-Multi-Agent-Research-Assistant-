import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from graph.workflow import build_research_graph
from graph.state import ResearchState
from agents.planner import plan_research
from agents.search import search_academic_sources
from rag.loader import load_pdf
from rag.chunker import chunk_document
from rag.vector_store import VectorStoreManager
from database.database import init_db, get_db
from database.models import (
    ResearchSession, Paper, PaperAnalysis, EvidenceItem,
    CitationItem, ContradictionItem, ResearchGapItem, ReportItem
)
from utils.logging import logger

init_db()
app = FastAPI(
    title="Evidentia AI API",
    description="Evidence-Driven Multi-Agent Research Assistant Backend",
    version="1.0.0"
)

# Shared LangGraph workflow instance
graph = build_research_graph()

from fastapi.responses import Response
from utils.pdf_generator import generate_research_pdf
from utils.formatting import generate_bibtex, parse_authors

class PlanRequest(BaseModel):
    question: str
    demo_mode: bool = False

class SearchRequest(BaseModel):
    queries: List[str]
    max_papers: int = 5
    include_web: bool = True
    demo_mode: bool = False
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    sources: Optional[List[str]] = None

class ResearchExecutionRequest(BaseModel):
    question: str
    max_papers: int = 5
    include_web: bool = True
    demo_mode: bool = False
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    sources: Optional[List[str]] = None
    provider: Optional[str] = "openai"
    model_name: Optional[str] = None

@app.get("/")
def root():
    return {
        "system": "Evidentia AI",
        "description": "Evidence-Driven Multi-Agent Research Assistant",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/research/plan")
def api_plan_research(req: PlanRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return plan_research(req.question, demo_mode=req.demo_mode)

@app.post("/api/research/search")
def api_search_sources(req: SearchRequest):
    return search_academic_sources(
        queries=req.queries,
        max_papers=req.max_papers,
        include_web=req.include_web,
        demo_mode=req.demo_mode
    )

@app.post("/api/research/execute")
def api_execute_research(req: ResearchExecutionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = str(uuid.uuid4())
    logger.info(f"API received execution request: '{req.question}' [Session: {session_id}]")

    initial_state: ResearchState = {
        "research_question": req.question,
        "session_id": session_id,
        "demo_mode": req.demo_mode,
        "max_papers": req.max_papers,
        "include_web": req.include_web,
        "search_iteration": 0,
        "year_min": req.year_min,
        "year_max": req.year_max,
        "sources": req.sources,
        "llm_provider": req.provider,
        "model_name": req.model_name,
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
        
        # Persist session to database
        with get_db() as db:
            session = ResearchSession(
                id=session_id,
                question=req.question,
                status="completed"
            )
            db.merge(session)

            for p in final_state.get("papers", []):
                authors_str = str(p.get("authors", []))
                paper_obj = Paper(
                    session_id=session_id,
                    title=p.get("title", ""),
                    authors=authors_str,
                    year=p.get("year"),
                    abstract=p.get("abstract", ""),
                    url=p.get("url", ""),
                    doi=p.get("doi", ""),
                    source=p.get("source", "")
                )
                db.add(paper_obj)

            if final_state.get("final_report"):
                rep_obj = ReportItem(
                    session_id=session_id,
                    markdown_content=final_state["final_report"]
                )
                db.add(rep_obj)

        return {
            "session_id": session_id,
            "status": "completed",
            "papers_count": len(final_state.get("papers", [])),
            "evidence_count": len(final_state.get("evidence", [])),
            "citations_count": len(final_state.get("citations", {})),
            "report": final_state.get("final_report", ""),
            "state": final_state
        }
    except Exception as e:
        logger.error(f"API execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/research/sessions")
def list_sessions():
    with get_db() as db:
        sessions = db.query(ResearchSession).order_by(ResearchSession.created_at.desc()).limit(20).all()
        return [
            {
                "id": s.id,
                "question": s.question,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None
            }
            for s in sessions
        ]

@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are permitted.")

    content = await file.read()
    extracted = load_pdf(content, filename=file.filename)
    if extracted["status"] == "error":
        raise HTTPException(status_code=400, detail=extracted["error"])

    chunks = chunk_document(extracted["pages"], filename=file.filename)
    return {
        "filename": extracted["filename"],
        "num_pages": extracted["num_pages"],
        "text_size_chars": extracted["text_size_chars"],
        "num_chunks": len(chunks),
        "status": "processed"
    }

@app.get("/api/research/{session_id}/pdf")
def get_session_pdf(session_id: str):
    """Generates and downloads publication-grade PDF report for a research session."""
    with get_db() as db:
        session = db.query(ResearchSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")

        papers_list = db.query(Paper).filter_by(session_id=session_id).all()
        analyses_list = db.query(PaperAnalysis).filter_by(session_id=session_id).all()
        evidence_list = db.query(EvidenceItem).filter_by(session_id=session_id).all()
        contradiction_list = db.query(ContradictionItem).filter_by(session_id=session_id).all()
        gap_list = db.query(ResearchGapItem).filter_by(session_id=session_id).all()
        report_item = db.query(ReportItem).filter_by(session_id=session_id).first()

        papers = [
            {
                "title": p.title,
                "authors": parse_authors(p.authors),
                "year": p.year,
                "url": p.url,
                "doi": p.doi,
                "source": p.source
            }
            for p in papers_list
        ]
        analyses = [
            {
                "paper": a.paper_title,
                "method": a.method,
                "dataset": a.dataset,
                "model": a.model,
                "results": a.results,
                "limitations": a.limitations
            }
            for a in analyses_list
        ]
        evidence = [
            {
                "claim": e.claim,
                "evidence": e.evidence,
                "source_title": e.source_title,
                "confidence": e.confidence
            }
            for e in evidence_list
        ]
        contradictions = [
            {
                "finding_a": c.finding_a,
                "source_a": c.source_a,
                "finding_b": c.finding_b,
                "source_b": c.source_b,
                "explanation": c.explanation
            }
            for c in contradiction_list
        ]
        gaps = [
            {
                "gap": g.gap,
                "reason": g.reason,
                "supporting_papers": parse_authors(g.supporting_papers)
            }
            for g in gap_list
        ]
        final_md = report_item.markdown_content if report_item else ""

        pdf_bytes = generate_research_pdf(
            research_question=session.question,
            papers=papers,
            paper_analyses=analyses,
            evidence=evidence,
            contradictions=contradictions,
            research_gaps=gaps,
            final_report_markdown=final_md,
            session_id=session_id
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=evidentia_report_{session_id[:8]}.pdf"}
        )

@app.get("/api/research/{session_id}/bibtex")
def get_session_bibtex(session_id: str):
    """Exports bibliography for a research session in standard BibTeX (.bib) format."""
    with get_db() as db:
        session = db.query(ResearchSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")

        papers_list = db.query(Paper).filter_by(session_id=session_id).all()
        papers = [
            {
                "title": p.title,
                "authors": parse_authors(p.authors),
                "year": p.year,
                "url": p.url,
                "doi": p.doi,
                "source": p.source
            }
            for p in papers_list
        ]
        bibtex_str = generate_bibtex(papers)
        return Response(
            content=bibtex_str,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=evidentia_{session_id[:8]}.bib"}
        )

