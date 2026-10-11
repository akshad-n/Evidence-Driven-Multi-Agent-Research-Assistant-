import pytest
from agents.critic import audit_research
from agents.report import generate_report
from data.sample_papers import get_demo_papers
from agents.analyzer import analyze_papers
from agents.evidence import extract_evidence

def test_critic_agent_audit():
    papers = get_demo_papers(limit=2)
    question = "What are the applications of Generative AI in healthcare?"
    analyses = analyze_papers(papers, demo_mode=True)
    evidence = extract_evidence(question, papers, demo_mode=True)

    critic_res = audit_research(question, evidence, analyses, demo_mode=True)

    assert "audit_results" in critic_res
    assert len(critic_res["audit_results"]) >= 1
    assert "status" in critic_res["audit_results"][0]
    assert critic_res["audit_results"][0]["status"] in ["verified", "partially_supported", "not_supported", "conflicting"]

    assert "contradictions" in critic_res
    assert isinstance(critic_res["contradictions"], list)

    assert "research_gaps" in critic_res
    assert len(critic_res["research_gaps"]) >= 1
    for gap in critic_res["research_gaps"]:
        assert "gap" in gap
        assert "potential research gap" in gap["gap"].lower()
        assert "reason" in gap

def test_report_generator_13_sections():
    papers = get_demo_papers(limit=2)
    question = "What are the applications of Generative AI in healthcare?"
    analyses = analyze_papers(papers, demo_mode=True)
    evidence = extract_evidence(question, papers, demo_mode=True)
    critic_res = audit_research(question, evidence, analyses, demo_mode=True)

    report = generate_report(
        research_question=question,
        papers=papers,
        paper_analyses=analyses,
        evidence=evidence,
        contradictions=critic_res["contradictions"],
        research_gaps=critic_res["research_gaps"],
        demo_mode=True
    )

    required_sections = [
        "## 1. Research Question",
        "## 2. Executive Summary",
        "## 3. Introduction",
        "## 4. Research Methodology",
        "## 5. Literature Findings",
        "## 6. Paper Comparison",
        "## 7. Evidence and Citations",
        "## 8. Contradictory Findings",
        "## 9. Limitations",
        "## 10. Potential Research Gaps",
        "## 11. Future Directions",
        "## 12. Conclusion",
        "## 13. References"
    ]

    for section in required_sections:
        assert section in report, f"Missing section: {section}"
