import pytest
import json
from agents.llm_provider import get_llm
from utils.validation import validate_research_question
from database.database import init_db, get_db
from database.models import ResearchSession

def test_validation():
    is_valid, msg = validate_research_question("")
    assert not is_valid
    assert "empty" in msg.lower()

    is_valid, msg = validate_research_question("What are the applications of Generative AI in healthcare?")
    assert is_valid

def test_llm_provider():
    llm = get_llm(demo_mode=True)
    res = llm.invoke("Research Planner for Question: What are the applications of Generative AI in healthcare?")
    assert res is not None
    assert len(res.content) > 10
    parsed = json.loads(res.content)
    assert "tasks" in parsed
    assert len(parsed["tasks"]) > 0

def test_database_session():
    init_db()
    with get_db() as db:
        session = ResearchSession(
            id="test-session-123",
            question="Test question",
            status="completed"
        )
        db.merge(session)

    with get_db() as db:
        fetched = db.query(ResearchSession).filter_by(id="test-session-123").first()
        assert fetched is not None
        assert fetched.question == "Test question"
