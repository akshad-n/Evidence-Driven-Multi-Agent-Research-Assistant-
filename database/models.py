from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def get_utc_now():
    return datetime.now(timezone.utc)

class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id = Column(String(36), primary_key=True)
    question = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_utc_now)
    status = Column(String(50), default="started")

    tasks = relationship("ResearchTask", back_populates="session", cascade="all, delete-orphan")
    papers = relationship("Paper", back_populates="session", cascade="all, delete-orphan")
    analyses = relationship("PaperAnalysis", back_populates="session", cascade="all, delete-orphan")
    evidences = relationship("EvidenceItem", back_populates="session", cascade="all, delete-orphan")
    citations = relationship("CitationItem", back_populates="session", cascade="all, delete-orphan")
    contradictions = relationship("ContradictionItem", back_populates="session", cascade="all, delete-orphan")
    gaps = relationship("ResearchGapItem", back_populates="session", cascade="all, delete-orphan")
    reports = relationship("ReportItem", back_populates="session", cascade="all, delete-orphan")


class ResearchTask(Base):
    __tablename__ = "research_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("research_sessions.id"), nullable=False)
    task_order = Column(Integer, default=0)
    description = Column(Text, nullable=False)
    status = Column(String(50), default="pending")

    session = relationship("ResearchSession", back_populates="tasks")


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("research_sessions.id"), nullable=False)
    title = Column(Text, nullable=False)
    authors = Column(Text, default="[]")  # JSON encoded list of authors
    year = Column(Integer, nullable=True)
    abstract = Column(Text, default="")
    url = Column(Text, default="")
    doi = Column(String(255), default="")
    source = Column(String(100), default="academic")

    session = relationship("ResearchSession", back_populates="papers")
    analyses = relationship("PaperAnalysis", back_populates="paper", cascade="all, delete-orphan")


class PaperAnalysis(Base):
    __tablename__ = "paper_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("research_sessions.id"), nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=True)
    paper_title = Column(Text, default="")
    method = Column(Text, default="Not specified in source")
    dataset = Column(Text, default="Not specified in source")
    model = Column(Text, default="Not specified in source")
    results = Column(Text, default="Not specified in source")
    advantages = Column(Text, default="Not specified in source")
    limitations = Column(Text, default="Not specified in source")
    future_work = Column(Text, default="Not specified in source")
    key_findings = Column(Text, default="[]")  # JSON encoded list

    session = relationship("ResearchSession", back_populates="analyses")
    paper = relationship("Paper", back_populates="analyses")


class EvidenceItem(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("research_sessions.id"), nullable=False)
    claim = Column(Text, nullable=False)
    evidence = Column(Text, nullable=False)
    source_title = Column(Text, default="")
    source_url = Column(Text, default="")
    doi = Column(String(255), default="")
    page = Column(String(50), default="Not available")
    section = Column(String(100), default="Not available")
    confidence = Column(Float, default=1.0)
    verification_status = Column(String(50), default="verified")

    session = relationship("ResearchSession", back_populates="evidences")


class CitationItem(Base):
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("research_sessions.id"), nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=True)
    citation_text = Column(Text, nullable=False)
    marker = Column(String(20), default="[1]")

    session = relationship("ResearchSession", back_populates="citations")


class ContradictionItem(Base):
    __tablename__ = "contradictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("research_sessions.id"), nullable=False)
    finding_a = Column(Text, nullable=False)
    source_a = Column(Text, nullable=False)
    finding_b = Column(Text, nullable=False)
    source_b = Column(Text, nullable=False)
    explanation = Column(Text, default="")

    session = relationship("ResearchSession", back_populates="contradictions")


class ResearchGapItem(Base):
    __tablename__ = "research_gaps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("research_sessions.id"), nullable=False)
    gap = Column(Text, nullable=False)
    supporting_papers = Column(Text, default="[]")  # JSON string
    reason = Column(Text, default="")

    session = relationship("ResearchSession", back_populates="gaps")


class ReportItem(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("research_sessions.id"), nullable=False)
    markdown_content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_utc_now)

    session = relationship("ResearchSession", back_populates="reports")
