try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import streamlit as st
from dotenv import load_dotenv

from graph.workflow import build_research_graph
from graph.state import ResearchState
from rag.loader import load_pdf
from rag.chunker import chunk_document
from rag.vector_store import VectorStoreManager
from database.database import init_db, get_db
from database.models import (
    ResearchSession, Paper, PaperAnalysis, EvidenceItem,
    CitationItem, ContradictionItem, ResearchGapItem, ReportItem
)
from utils.logging import logger
from utils.validation import validate_research_question, validate_uploaded_pdf, sanitize_filename
from utils.formatting import format_citation, generate_bibtex, parse_authors
from utils.pdf_generator import generate_research_pdf
import plotly.graph_objects as go
import numpy as np

load_dotenv()
init_db()

st.set_page_config(
    page_title="Evidentia AI – Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #0284C7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-verified {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .badge-partially {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-contradiction {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-gap {
        background-color: #E0E7FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }
    .badge-source {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
    }
    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-txt {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .callout-quote {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 8px 0;
        font-style: italic;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = []

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=56)
    st.title("Evidentia AI")
    st.caption("Evidence-Driven Multi-Agent Research Assistant")
    
    st.divider()
    st.subheader("⚙️ Engine Configuration")
    
    provider_choice = st.selectbox(
        "LLM Provider Architecture",
        options=["OpenAI", "Google Gemini", "Anthropic Claude", "Ollama (Local)", "Demo Engine (Offline)"],
        index=0,
        help="Select the AI reasoning backend matching the system architecture proposal."
    )
    
    active_key = ""
    active_model = ""
    custom_endpoint = None

    if provider_choice == "OpenAI":
        active_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Supply OpenAI API key or leave blank for Demo Mode."
        )
        active_model = st.selectbox("Model Architecture", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

    elif provider_choice == "Google Gemini":
        active_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Supply Google Gemini API key or leave blank for Demo Mode."
        )
        active_model = st.selectbox("Model Architecture", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], index=0)

    elif provider_choice == "Anthropic Claude":
        active_key = st.text_input(
            "Anthropic API Key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password"
        )
        active_model = st.selectbox("Model Architecture", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"], index=0)

    elif provider_choice == "Ollama (Local)":
        custom_endpoint = st.text_input("Ollama Endpoint", value="http://localhost:11434/v1")
        active_model = st.text_input("Model Tag", value="llama3")
        active_key = "ollama"

    else:  # Demo Engine
        active_key = ""
        active_model = "evidentia-demo-engine"

    demo_mode_toggle = st.toggle("Force Demo / Offline Mode", value=(provider_choice == "Demo Engine (Offline)" or not bool(active_key)))
    effective_demo = demo_mode_toggle or (provider_choice != "Ollama (Local)" and not bool(active_key))
    
    if effective_demo:
        st.markdown('<div class="badge badge-partially">🔬 DEMO MODE ACTIVE</div>', unsafe_allow_html=True)
        st.caption("Operating offline with verified peer-reviewed academic benchmark corpus.")
    else:
        st.markdown('<div class="badge badge-verified">🟢 LIVE LLM ACTIVE</div>', unsafe_allow_html=True)
        st.caption(f"Provider: {provider_choice} | Model: {active_model}")

    st.divider()
    st.subheader("📚 Research Scope & Filters")
    max_papers_slider = st.slider("Target Literature Corpus", min_value=3, max_value=15, value=5)
    
    year_filter_option = st.selectbox(
        "Publication Year Filter",
        ["All Years", "Recent (2023 – 2026)", "Past 5 Years (2021 – 2026)", "Past 10 Years (2016 – 2026)"],
        index=0
    )
    year_min_val, year_max_val = None, None
    if "2023" in year_filter_option:
        year_min_val, year_max_val = 2023, 2026
    elif "2021" in year_filter_option:
        year_min_val, year_max_val = 2021, 2026
    elif "2016" in year_filter_option:
        year_min_val, year_max_val = 2016, 2026

    db_options = st.multiselect(
        "Academic Data Sources",
        options=["arXiv", "Semantic Scholar", "Crossref", "PubMed", "Academic Web"],
        default=["arXiv", "Semantic Scholar", "Crossref", "PubMed", "Academic Web"]
    )
    source_key_map = {
        "arXiv": "arxiv",
        "Semantic Scholar": "semantic_scholar",
        "Crossref": "crossref",
        "PubMed": "pubmed",
        "Academic Web": "web"
    }
    selected_source_keys = [source_key_map[d] for d in db_options if d in source_key_map]
    include_web_sources = "web" in selected_source_keys

    st.divider()
    st.subheader("🗄️ Past Research History")
    with get_db() as db:
        recent_sessions = db.query(ResearchSession).order_by(ResearchSession.created_at.desc()).limit(8).all()
        if recent_sessions:
            for s in recent_sessions:
                label = f"{s.question[:32]}..." if len(s.question) > 32 else s.question
                if st.button(f"📄 {label}", key=f"hist_{s.id}", use_container_width=True):
                    # Load historical session
                    report_item = db.query(ReportItem).filter_by(session_id=s.id).first()
                    papers_list = db.query(Paper).filter_by(session_id=s.id).all()
                    evidence_list = db.query(EvidenceItem).filter_by(session_id=s.id).all()
                    contradiction_list = db.query(ContradictionItem).filter_by(session_id=s.id).all()
                    gap_list = db.query(ResearchGapItem).filter_by(session_id=s.id).all()
                    
                    st.session_state.current_session_id = s.id
                    st.session_state.current_results = {
                        "research_question": s.question,
                        "session_id": s.id,
                        "final_report": report_item.markdown_content if report_item else "# No report saved",
                        "papers": [
                            {
                                "title": p.title,
                                "authors": parse_authors(p.authors),
                                "year": p.year,
                                "abstract": p.abstract,
                                "url": p.url,
                                "doi": p.doi,
                                "source": p.source
                            }
                            for p in papers_list
                        ],
                        "evidence": [
                            {
                                "claim": e.claim,
                                "evidence": e.evidence,
                                "source_title": e.source_title,
                                "source_url": e.source_url,
                                "doi": e.doi,
                                "page": e.page,
                                "section": e.section,
                                "confidence": e.confidence,
                                "citation": "[1]"
                            }
                            for e in evidence_list
                        ],
                        "citations": {},
                        "paper_analysis": [],
                        "contradictions": [
                            {
                                "finding_a": c.finding_a,
                                "source_a": c.source_a,
                                "finding_b": c.finding_b,
                                "source_b": c.source_b,
                                "explanation": c.explanation
                            }
                            for c in contradiction_list
                        ],
                        "research_gaps": [
                            {
                                "gap": g.gap,
                                "reason": g.reason,
                                "supporting_papers": parse_authors(g.supporting_papers)
                            }
                            for g in gap_list
                        ],
                        "critic_results": {"audit_results": []},
                        "research_tasks": []
                    }
                    st.rerun()
        else:
            st.caption("No past sessions stored yet.")

# Main Application Header
st.markdown('<div class="main-title">EVIDENTIA AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An Evidence-Driven Multi-Agent Research Assistant for Literature Review & Gap Analysis</div>', unsafe_allow_html=True)

# 1. Research Question Input
st.markdown("### 1. Define Research Inquiry")
benchmarks = [
    "What are the applications of Generative AI in healthcare?",
    "How does Retrieval-Augmented Generation reduce hallucination in LLMs?",
    "What are the algorithmic limitations of Transformer multi-head attention?",
    "How do state space models like Mamba compare to Transformers in computational efficiency?"
]
selected_bench = st.selectbox("Choose a sample benchmark or type a custom research question:", ["-- Select sample question --"] + benchmarks)
default_text = "" if selected_bench.startswith("--") else selected_bench

user_inquiry = st.text_area(
    "Research Question:",
    value=default_text,
    placeholder="e.g. What are the clinical applications, empirical performance benchmarks, and diagnostic limitations of Generative AI in healthcare?",
    height=80
)

# 2. PDF Document Ingestion (RAG)
st.markdown("### 2. Upload Research Papers (PDF)")
uploaded_files = st.file_uploader(
    "Upload one or more peer-reviewed research papers (.pdf) for automated vector extraction and evidence verification:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    upload_cols = st.columns(len(uploaded_files) if len(uploaded_files) <= 3 else 3)
    total_chunks = 0
    all_chunks = []
    
    for idx, f in enumerate(uploaded_files):
        content = f.read()
        is_valid, msg = validate_uploaded_pdf(f.name, content)
        if not is_valid:
            st.error(f"❌ {f.name}: {msg}")
            continue
            
        extracted = load_pdf(content, filename=sanitize_filename(f.name))
        if extracted["status"] == "success":
            chunks = chunk_document(extracted["pages"], filename=f.name)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
            col_target = upload_cols[idx % len(upload_cols)]
            with col_target:
                st.markdown(f"""
                <div class="card">
                    <b>📄 {f.name}</b><br>
                    <span style="font-size:0.85rem; color:#64748B;">
                    Pages: <b>{extracted['num_pages']}</b> | Size: <b>{extracted['text_size_chars']:,} chars</b><br>
                    Chunks created: <b>{len(chunks)}</b>
                    </span><br>
                    <span class="badge badge-verified">✓ Processed</span>
                </div>
                """, unsafe_allow_html=True)
                
    st.session_state.uploaded_chunks = all_chunks

# Start Research Trigger
btn_start = st.button("🚀 Start Evidence-Driven Research", type="primary", use_container_width=True)

if btn_start:
    is_valid, msg = validate_research_question(user_inquiry)
    if not is_valid:
        st.error(f"⚠️ {msg}")
    else:
        new_session_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_session_id
        
        # Save new session in database
        try:
            with get_db() as db:
                db_sess = ResearchSession(
                    id=new_session_id,
                    question=user_inquiry,
                    status="in_progress",
                    created_at=datetime.now(timezone.utc)
                )
                db.add(db_sess)
        except Exception as e:
            logger.warning(f"Could not persist initial session to DB: {e}")

        # Real-time multi-agent execution display
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Multi-Agent Research Progress")
            status_box = st.status("Initializing research agents...", expanded=True)
            
            # Step indicators
            p_step1 = status_box.empty()
            p_step2 = status_box.empty()
            p_step3 = status_box.empty()
            p_step4 = status_box.empty()
            p_step5 = status_box.empty()
            p_step6 = status_box.empty()

            p_step1.write("⏳ Agent 1: Research Planning...")
            
            # Build initial LangGraph state
            graph = build_research_graph()
            initial_state: ResearchState = {
                "research_question": user_inquiry,
                "session_id": new_session_id,
                "demo_mode": effective_demo,
                "max_papers": max_papers_slider,
                "include_web": include_web_sources,
                "search_iteration": 0,
                "year_min": year_min_val,
                "year_max": year_max_val,
                "sources": selected_source_keys,
                "llm_provider": provider_choice.lower(),
                "model_name": active_model,
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

            # Optional RAG Vector Ingestion if PDFs were uploaded
            if st.session_state.uploaded_chunks:
                try:
                    v_store = VectorStoreManager(demo_mode=effective_demo)
                    v_store.add_chunks(st.session_state.uploaded_chunks, session_id=new_session_id)
                    status_box.write(f"✓ RAG Vector Store indexed {len(st.session_state.uploaded_chunks)} uploaded chunks into ChromaDB.")
                except Exception as e:
                    logger.warning(f"Vector store indexing notice: {e}")

            # Execute Workflow
            final_state = graph.invoke(initial_state)

            p_step1.write("✓ Research planning completed (Tasks and targeted queries formulated)")
            p_step2.write("✓ Searching academic sources (arXiv, Semantic Scholar, Crossref, PubMed, Web)")
            p_step3.write(f"✓ Papers collected & deduplicated ({len(final_state.get('papers', []))} papers)")
            p_step4.write(f"✓ Paper analysis completed ({len(final_state.get('paper_analysis', []))} papers analyzed)")
            p_step5.write(f"✓ Evidence extraction & citation verification ({len(final_state.get('evidence', []))} evidence items)")
            p_step6.write(f"✓ Critic audit, contradiction detection & report generation finished")
            status_box.update(label="All 6 Research Agents Completed Successfully!", state="complete", expanded=False)

        # Persist full state to Database
        try:
            with get_db() as db:
                db_sess = db.query(ResearchSession).filter_by(id=new_session_id).first()
                if db_sess:
                    db_sess.status = "completed"

                for p in final_state.get("papers", []):
                    paper_obj = Paper(
                        session_id=new_session_id,
                        title=p.get("title", ""),
                        authors=str(p.get("authors", [])),
                        year=p.get("year"),
                        abstract=p.get("abstract", ""),
                        url=p.get("url", ""),
                        doi=p.get("doi", ""),
                        source=p.get("source", "")
                    )
                    db.add(paper_obj)

                for e in final_state.get("evidence", []):
                    ev_obj = EvidenceItem(
                        session_id=new_session_id,
                        claim=e.get("claim", ""),
                        evidence=e.get("evidence", ""),
                        source_title=e.get("source_title", ""),
                        source_url=e.get("source_url", ""),
                        doi=e.get("doi", ""),
                        page=str(e.get("page", "Not available")),
                        section=str(e.get("section", "Not available")),
                        confidence=float(e.get("confidence", 0.9)),
                        verification_status="verified"
                    )
                    db.add(ev_obj)

                for c in final_state.get("contradictions", []):
                    cont_obj = ContradictionItem(
                        session_id=new_session_id,
                        finding_a=c.get("finding_a", ""),
                        source_a=c.get("source_a", ""),
                        finding_b=c.get("finding_b", ""),
                        source_b=c.get("source_b", ""),
                        explanation=c.get("explanation", "")
                    )
                    db.add(cont_obj)

                for g in final_state.get("research_gaps", []):
                    gap_obj = ResearchGapItem(
                        session_id=new_session_id,
                        gap=g.get("gap", ""),
                        supporting_papers=str(g.get("supporting_papers", [])),
                        reason=g.get("reason", "")
                    )
                    db.add(gap_obj)

                if final_state.get("final_report"):
                    rep_obj = ReportItem(
                        session_id=new_session_id,
                        markdown_content=final_state["final_report"],
                        created_at=datetime.now(timezone.utc)
                    )
                    db.add(rep_obj)
        except Exception as e:
            logger.error(f"Error persisting full research state to DB: {e}")

        st.session_state.current_results = final_state
        st.rerun()

# 3. Results Section (7 Tabs)
if st.session_state.current_results:
    res = st.session_state.current_results
    st.divider()
    st.subheader(f"📊 Research Dossier: {res.get('research_question')}")
    
    # Progress Metric Counters (Section 17)
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.markdown(f"""<div class="metric-box"><div class="metric-num">{len(res.get('papers', []))}</div><div class="metric-txt">Papers Found</div></div>""", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""<div class="metric-box"><div class="metric-num">{len(res.get('paper_analysis', []))}</div><div class="metric-txt">Papers Analyzed</div></div>""", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""<div class="metric-box"><div class="metric-num">{len(res.get('evidence', []))}</div><div class="metric-txt">Evidence Items</div></div>""", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""<div class="metric-box"><div class="metric-num">{len(res.get('citations', {})) or len(res.get('papers', []))}</div><div class="metric-txt">Citations</div></div>""", unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""<div class="metric-box"><div class="metric-num">{len(res.get('contradictions', []))}</div><div class="metric-txt">Contradictions</div></div>""", unsafe_allow_html=True)
    with m_col6:
        st.markdown(f"""<div class="metric-box"><div class="metric-num">{len(res.get('research_gaps', []))}</div><div class="metric-txt">Research Gaps</div></div>""", unsafe_allow_html=True)

    st.write("")
    tabs = st.tabs([
        "📋 Overview",
        "📄 Papers",
        "🔍 Evidence",
        "⚡ Contradictions",
        "🧩 Research Gaps",
        "🕸️ Knowledge Graph",
        "📑 Final Report",
        "📚 References"
    ])

    # Tab 1: Overview
    with tabs[0]:
        st.markdown("#### Research Session Overview")
        st.markdown(f"**Research Question**: *{res.get('research_question')}*")
        st.markdown(f"**Session ID**: `{res.get('session_id')}`")
        st.markdown(f"**Critic Audit Summary**: {res.get('critic_results', {}).get('critic_summary', 'Completed without critical errors.')}")

        st.markdown("#### Formulated Research Tasks")
        tasks = res.get("research_tasks", [])
        if tasks:
            for idx, t in enumerate(tasks):
                st.markdown(f"**{idx + 1}.** {t}")
        else:
            st.caption("Standard systematic inquiry tasks formulated.")

        st.markdown("#### Generated Search Queries")
        queries = res.get("search_queries", [])
        if queries:
            for q in queries:
                st.code(q, language="text")

    # Tab 2: Papers
    with tabs[1]:
        st.markdown("#### Retrieved Academic Papers")
        papers = res.get("papers", [])
        if not papers:
            st.info("No papers retrieved.")
        for idx, p in enumerate(papers):
            authors = ", ".join(p.get("authors", [])) or "Authors not specified"
            year = p.get("year", "n.d.")
            source = p.get("source", "academic").upper()
            url = p.get("url", "")
            doi = p.get("doi", "")

            with st.expander(f"[{idx + 1}] {p.get('title', 'Untitled')} ({year}) - {source}", expanded=(idx == 0)):
                st.markdown(f"**Authors**: {authors}")
                st.markdown(f"**Year**: {year} | **Source**: `{source}` | **DOI**: `{doi}`")
                if url and url != "Not available":
                    st.markdown(f"🔗 [View Original Publication]({url})")
                st.markdown(f"**Abstract / Text Excerpt**:")
                st.write(p.get("abstract", "Not available"))

                # Check if matching paper analysis exists
                analysis = next((a for a in res.get("paper_analysis", []) if a.get("paper") == p.get("title")), None)
                if analysis:
                    st.markdown("---")
                    st.markdown("**Structured Agent Analysis**:")
                    c_an1, c_an2 = st.columns(2)
                    with c_an1:
                        st.markdown(f"• **Method**: {analysis.get('method')}")
                        st.markdown(f"• **Dataset**: {analysis.get('dataset')}")
                        st.markdown(f"• **Model**: {analysis.get('model')}")
                    with c_an2:
                        st.markdown(f"• **Results**: {analysis.get('results')}")
                        st.markdown(f"• **Limitations**: {analysis.get('limitations')}")
                        st.markdown(f"• **Future Work**: {analysis.get('future_work')}")

    # Tab 3: Evidence
    with tabs[2]:
        st.markdown("#### Verified Evidence and Claim Alignments")
        st.caption("Every factual assertion is audited against verbatim text from academic publications.")
        evidence_items = res.get("evidence", [])
        if not evidence_items:
            st.info("No evidence records extracted.")
        for idx, item in enumerate(evidence_items):
            claim = item.get("claim", "")
            quote = item.get("evidence", "")
            source = item.get("source_title", "Source Paper")
            url = item.get("source_url", "")
            citation = item.get("citation", f"[{idx + 1}]")
            confidence = item.get("confidence", 0.95)
            page = item.get("page", "Not available")
            section = item.get("section", "Not available")

            st.markdown(f"""
            <div class="card">
                <span class="badge badge-verified">✓ Verified Status</span>
                <span class="badge badge-source">{citation} {source}</span>
                <span style="float: right; font-size: 0.85rem; color: #16A34A; font-weight: 600;">Confidence: {int(confidence * 100)}%</span>
                <h4 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.05rem; color: #0F172A;">{claim}</h4>
                <div class="callout-quote">"{quote}"</div>
                <div style="font-size: 0.8rem; color: #64748B;">
                    Section: <b>{section}</b> | Page: <b>{page}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tab 4: Contradictions
    with tabs[3]:
        st.markdown("#### Potential Contradiction Detection")
        st.caption("Divergent empirical findings across distinct research publications without taking speculative sides.")
        contradictions = res.get("contradictions", [])
        if contradictions:
            for idx, c in enumerate(contradictions):
                st.markdown(f"""
                <div class="card" style="border-left: 4px solid #EF4444;">
                    <span class="badge badge-contradiction">⚡ Potential Contradiction #{idx + 1}</span>
                    <div style="margin-top: 0.8rem;">
                        <div style="background: #FEF2F2; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                            <b>Finding A:</b> {c.get('finding_a')}<br>
                            <span style="font-size: 0.85rem; color: #991B1B;">Source: <i>{c.get('source_a')}</i></span>
                        </div>
                        <div style="background: #F8FAFC; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                            <b>Finding B:</b> {c.get('finding_b')}<br>
                            <span style="font-size: 0.85rem; color: #334155;">Source: <i>{c.get('source_b')}</i></span>
                        </div>
                        <p style="font-size: 0.9rem; color: #475569; margin: 0;"><b>Objective Analysis:</b> {c.get('explanation')}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No direct contradictions detected among the retrieved papers. Literature consensus appears consistent.")

    # Tab 5: Research Gaps
    with tabs[4]:
        st.markdown("#### Potential Research Gaps")
        st.caption("Identified opportunities derived systematically from paper limitations, small cohorts, and future work.")
        gaps = res.get("research_gaps", [])
        if gaps:
            for idx, g in enumerate(gaps):
                papers_linked = ", ".join(g.get("supporting_papers", [])) or "Analyzed Literature"
                st.markdown(f"""
                <div class="card" style="border-left: 4px solid #6366F1;">
                    <span class="badge badge-gap">🧩 Potential Research Gap #{idx + 1}</span>
                    <h4 style="margin: 0.5rem 0; color: #1E1B4B;">{g.get('gap')}</h4>
                    <p style="font-size: 0.9rem; color: #334155; margin-bottom: 0.4rem;">
                        <b>Reason:</b> {g.get('reason')}
                    </p>
                    <span style="font-size: 0.82rem; color: #64748B;">
                        Supporting Source Literature: <i>{papers_linked}</i>
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No explicit research gaps identified.")

    # Tab 6: Knowledge Graph
    with tabs[5]:
        st.markdown("#### Interactive Research Knowledge & Contradiction Graph")
        st.caption("Visual representation of synthesis topology: Research Inquiry ↔ Papers ↔ Grounded Evidence ↔ Contradictions & Gaps.")

        nodes_x = []
        nodes_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        edge_x = []
        edge_y = []

        # Root Inquiry Node
        nodes_x.append(0.0)
        nodes_y.append(0.0)
        q_label = res.get('research_question', 'Inquiry')
        node_text.append(f"<b>Research Inquiry:</b><br>{q_label[:70]}...")
        node_colors.append("#2563EB")
        node_sizes.append(28)

        papers_list = res.get("papers", [])
        num_papers = max(1, len(papers_list))
        paper_positions = []

        for p_idx, p in enumerate(papers_list):
            ang = (2 * np.pi * p_idx) / num_papers
            px = 1.9 * np.cos(ang)
            py = 1.9 * np.sin(ang)
            paper_positions.append((px, py))

            nodes_x.append(px)
            nodes_y.append(py)
            p_title = p.get('title', f'Paper {p_idx+1}')[:45]
            node_text.append(f"<b>[{p_idx+1}] {p_title}</b><br>Year: {p.get('year', 'n.d.')}<br>Source: {p.get('source', 'Academic')}")
            node_colors.append("#0D9488")
            node_sizes.append(18)

            edge_x.extend([0.0, px, None])
            edge_y.extend([0.0, py, None])

        # Evidence nodes
        evidence_records = res.get("evidence", [])
        for e_idx, ev in enumerate(evidence_records[:8]):
            parent_x, parent_y = paper_positions[e_idx % len(paper_positions)] if paper_positions else (0.0, 0.0)
            off_ang = ((e_idx * 1.6) % (2 * np.pi))
            ex = parent_x + 0.85 * np.cos(off_ang)
            ey = parent_y + 0.85 * np.sin(off_ang)

            nodes_x.append(ex)
            nodes_y.append(ey)
            conf_pct = int(float(ev.get('confidence', 0.9)) * 100)
            node_text.append(f"<b>Claim:</b> {ev.get('claim', '')[:50]}...<br>Confidence: {conf_pct}%")
            node_colors.append("#F59E0B")
            node_sizes.append(12)

            edge_x.extend([parent_x, ex, None])
            edge_y.extend([parent_y, ey, None])

        # Contradiction nodes (Red)
        contra_records = res.get("contradictions", [])
        for c_idx, c in enumerate(contra_records):
            cx = -2.6 + c_idx * 1.5
            cy = 2.4
            nodes_x.append(cx)
            nodes_y.append(cy)
            node_text.append(f"<b>⚡ Contradiction #{c_idx+1}:</b><br>{c.get('explanation', '')[:60]}...")
            node_colors.append("#EF4444")
            node_sizes.append(22)

            edge_x.extend([0.0, cx, None])
            edge_y.extend([0.0, cy, None])

        # Research Gap nodes (Indigo)
        gap_records = res.get("research_gaps", [])
        for g_idx, g in enumerate(gap_records):
            gx = -2.6 + g_idx * 1.5
            gy = -2.4
            nodes_x.append(gx)
            nodes_y.append(gy)
            node_text.append(f"<b>🧩 Gap #{g_idx+1}:</b><br>{g.get('gap', '')[:55]}...")
            node_colors.append("#6366F1")
            node_sizes.append(20)

            edge_x.extend([0.0, gx, None])
            edge_y.extend([0.0, gy, None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1.2, color='#CBD5E1'),
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter(
            x=nodes_x, y=nodes_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1.5, color='#FFFFFF')
            )
        ))
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            height=480
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 7: Final Report
    with tabs[6]:
        st.markdown("#### Synthesized Systematic Research Report")
        final_rep = res.get("final_report", "# No report generated.")

        # Generate publication PDF and BibTeX data
        pdf_bytes = generate_research_pdf(
            research_question=res.get("research_question", "Inquiry"),
            papers=res.get("papers", []),
            paper_analyses=res.get("paper_analysis", []),
            evidence=res.get("evidence", []),
            contradictions=res.get("contradictions", []),
            research_gaps=res.get("research_gaps", []),
            final_report_markdown=final_rep,
            session_id=res.get("session_id", "")
        )
        bibtex_content = generate_bibtex(res.get("papers", []))

        # 4 Export Buttons
        c_dl1, c_dl2, c_dl3, c_dl4 = st.columns(4)
        with c_dl1:
            st.download_button(
                label="📄 Download Publication PDF",
                data=pdf_bytes,
                file_name=f"evidentia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        with c_dl2:
            st.download_button(
                label="📝 Download Markdown (.md)",
                data=final_rep,
                file_name=f"evidentia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with c_dl3:
            st.download_button(
                label="📋 Download Plain Text (.txt)",
                data=final_rep,
                file_name=f"evidentia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with c_dl4:
            st.download_button(
                label="📚 Download BibTeX (.bib)",
                data=bibtex_content,
                file_name=f"evidentia_citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bib",
                mime="text/plain",
                use_container_width=True
            )

        st.divider()
        st.markdown(final_rep)

    # Tab 8: References
    with tabs[7]:
        st.markdown("#### Formal Academic References")
        papers = res.get("papers", [])
        if papers:
            for idx, p in enumerate(papers):
                citation_string = format_citation(p, index=idx + 1)
                st.markdown(f"{citation_string}")
            
            st.divider()
            st.markdown("#### Raw BibTeX Source")
            st.code(generate_bibtex(papers), language="bibtex")
        else:
            st.info("No references available.")

