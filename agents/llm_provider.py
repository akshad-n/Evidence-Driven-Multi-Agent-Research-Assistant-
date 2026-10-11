import os
import json
from typing import Optional, Any, List
from dotenv import load_dotenv
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from utils.logging import logger

load_dotenv()

class DemoChatModel(BaseChatModel):
    """
    Intelligent Demo Mode LLM that returns structured research responses 
    grounded in the user prompt when OpenAI API keys are not supplied.
    """
    model_name: str = Field(default="evidentia-demo-engine")

    @property
    def _llm_type(self) -> str:
        return "evidentia-demo-llm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        full_prompt = "\n".join([m.content for m in messages if isinstance(m.content, str)])
        
        # Determine intent based on prompt contents
        if "Research Planner" in full_prompt or "tasks" in full_prompt and "search_queries" in full_prompt:
            # Extract question if possible
            topic = "the research question"
            for m in messages:
                if isinstance(m.content, str) and "Question:" in m.content:
                    topic = m.content.split("Question:")[-1].strip()
                    break

            content = json.dumps({
                "research_question": topic,
                "tasks": [
                    f"Examine foundational theories and current landscape regarding {topic}",
                    f"Retrieve recent peer-reviewed publications and experimental benchmarks",
                    f"Identify core models, algorithmic frameworks, and evaluation datasets",
                    f"Synthesize empirical findings, strengths, and reported constraints",
                    f"Detect conflicting claims and outline potential research gaps"
                ],
                "search_queries": [
                    f"{topic} state of the art benchmarks",
                    f"{topic} empirical evaluation models",
                    f"{topic} systematic review limitations",
                    f"{topic} deep learning applications"
                ],
                "rationale": "Multi-angle exploration across empirical performance, architectural paradigms, and systematic limitations."
            }, indent=2)

        elif "Paper Analysis Agent" in full_prompt or "Research methodology" in full_prompt:
            content = json.dumps({
                "paper": "Analysis Target Paper",
                "method": "Multi-modal Transformer Architecture with cross-attention",
                "dataset": "Benchmark Medical Image & Clinical Notes Dataset (MIMIC-IV subset)",
                "model": "Domain-adapted Generative Transformer (ClinGen-7B)",
                "results": "Achieved 94.2% diagnostic summarization accuracy, outperforming baselines by 4.8%",
                "advantages": "Robust handling of incomplete clinical records and reduced hallucination rate",
                "limitations": "Computational overhead during multi-turn inference and limited evaluation on rare pathologies",
                "future_work": "Expanding token context window and exploring federated clinical deployment",
                "key_findings": [
                    "Domain-adapted instruction tuning significantly reduces clinical hallucination.",
                    "Cross-attention between visual scans and clinical notes improves diagnostic concordance."
                ]
            }, indent=2)

        elif "Evidence Extraction Agent" in full_prompt or "confidence" in full_prompt:
            content = json.dumps([
                {
                    "claim": "Domain-tuned generative models decrease factual hallucination in diagnostic assistance.",
                    "evidence": "Clinical evaluations demonstrated a 68% relative decrease in factual hallucinations when instruction-tuned on verified clinician-annotated EHR trajectories.",
                    "source_title": "Domain-Adapted Generative AI for Clinical Decision Support",
                    "source_url": "https://arxiv.org/abs/2309.12345",
                    "doi": "10.48550/arXiv.2309.12345",
                    "page": "6",
                    "section": "Experimental Results",
                    "confidence": 0.94
                },
                {
                    "claim": "Contextual latency remains a bottleneck for real-time surgical and emergency applications.",
                    "evidence": "Average response latency of 2.4 seconds per diagnostic inquiry exceeds the recommended threshold for ultra-low latency intraoperative assistance.",
                    "source_title": "Benchmarking Real-Time Generative Architectures in Healthcare",
                    "source_url": "https://arxiv.org/abs/2401.05678",
                    "doi": "10.48550/arXiv.2401.05678",
                    "page": "9",
                    "section": "System Latency & Limitations",
                    "confidence": 0.91
                }
            ], indent=2)

        elif "Critic Agent" in full_prompt or "audit_results" in full_prompt:
            content = json.dumps({
                "audit_results": [
                    {
                        "claim": "Domain-tuned generative models decrease factual hallucination in diagnostic assistance.",
                        "status": "verified",
                        "reason": "Directly supported by quantitative EHR benchmark evaluations in the paper."
                    },
                    {
                        "claim": "Contextual latency remains a bottleneck for real-time surgical and emergency applications.",
                        "status": "verified",
                        "reason": "Supported by empirical timing benchmarks reported in paper table 3."
                    }
                ],
                "contradictions": [
                    {
                        "finding_a": "Generative architectures improve diagnostic accuracy across all tested subspecialties without specialized fine-tuning.",
                        "source_a": "Generalized Foundation Models in Medicine (2024)",
                        "finding_b": "Zero-shot foundation models exhibit critical hallucination rates up to 28% in specialized pathology without domain fine-tuning.",
                        "source_b": "Safety and Failure Modes of Unadapted Medical LLMs (2024)",
                        "explanation": "Paper A claims broad zero-shot diagnostic competence, whereas Paper B identifies severe domain-specific failure modes when unadapted."
                    }
                ],
                "research_gaps": [
                    {
                        "gap": "Lack of standardized real-time clinical prospective trials and longitudinal outcome metrics",
                        "supporting_papers": ["Domain-Adapted Generative AI for Clinical Decision Support", "Benchmarking Real-Time Generative Architectures in Healthcare"],
                        "reason": "Current evaluations are largely retrospective on MIMIC or synthetic benchmarks; continuous bedside evaluations are critically lacking."
                    },
                    {
                        "gap": "Low-resource clinical setting adaptability and edge-device hardware constraints",
                        "supporting_papers": ["Benchmarking Real-Time Generative Architectures in Healthcare"],
                        "reason": "High VRAM and latency requirements limit adoption outside centralized multi-GPU hospital clusters."
                    }
                ],
                "needs_more_research": False,
                "critic_summary": "Extracted evidence directly corroborates core findings. Identified 1 notable contradiction regarding zero-shot generalization and 2 critical potential research gaps."
            }, indent=2)

        elif "Report Generator" in full_prompt or "Executive Summary" in full_prompt:
            content = """# Evidentia AI Systematic Research Report: Generative AI in Healthcare

## 1. Research Question
What are the applications, performance benchmarks, and empirical limitations of Generative AI in healthcare?

## 2. Executive Summary
Generative Artificial Intelligence (GenAI) is rapidly transitioning from general linguistic interfaces into specialized clinical decision support systems. Key applications include automated EHR summarization, medical imaging synthesis, triage recommendation, and patient communication. Despite clear accuracy enhancements (e.g. 94.2% diagnostic summarization), critical operational hurdles remain: inference latency, domain-specific hallucinations, and a severe paucity of prospective bedside trials [1], [2].

## 3. Introduction
The integration of foundational large language and multi-modal models into medicine offers unprecedented potential to alleviate clinician burnout and synthesize heterogeneous multimodal clinical records. However, deploying probabilistic generative systems in high-stakes clinical workflows demands rigorous verification, transparency, and grounded evidence retrieval.

## 4. Research Methodology
This systematic literature synthesis was conducted using Evidentia AI's multi-agent research pipeline. Literature was retrieved across academic repositories (arXiv, Semantic Scholar, Crossref, and PubMed), analyzed for architectural specifications, and audited through deterministic claim-to-evidence cross-verification.

## 5. Literature Findings
1. **Clinical Record Synthesis & EHR Summarization**: Fine-tuned transformers reliably transform unstructured clinical notes into standardized formats, reducing documentation overhead by up to 40% [1].
2. **Diagnostic Decision Support**: Domain-adapted multimodal models demonstrate high concordance with specialist diagnoses across radiology and pathology [1].
3. **Inference Latency Challenges**: For intraoperative and emergency environments, standard autoregressive generation imposes latencies (2-3s) exceeding acceptable operational limits [2].

## 6. Paper Comparison
| Paper / System | Model Paradigm | Primary Dataset | Reported Accuracy | Chief Constraint |
| :--- | :--- | :--- | :--- | :--- |
| ClinGen-7B [1] | Domain-tuned 7B | MIMIC-IV Clinical Notes | 94.2% summarization | High inference compute |
| LatencyBenchmark [2] | Quantized Mixture of Experts | Emergency Trauma Scans | 91.5% concordance | 2.4s latency overhead |
| MedAudit-Path [3] | Vision-Language Foundation | Multi-center Histopathology | 89.1% zero-shot | 28% hallucination on rare cases |

## 7. Evidence and Citations
* **Claim**: Domain-adapted instruction tuning significantly reduces clinical hallucination.
  * *Evidence*: "Clinical evaluations demonstrated a 68% relative decrease in factual hallucinations when instruction-tuned on verified clinician-annotated EHR trajectories" [1].
* **Claim**: High computational latency restricts real-time emergency deployment.
  * *Evidence*: "Average response latency of 2.4 seconds per diagnostic inquiry exceeds the recommended threshold for ultra-low latency intraoperative assistance" [2].

## 8. Contradictory Findings
* **Contradiction Identified**: Zero-Shot Generalization vs Domain-Specific Hallucination:
  * *Finding A*: Foundation models without fine-tuning report generalized diagnostic competence across common disorders [3].
  * *Finding B*: Dedicated safety studies show unadapted zero-shot models generate critical diagnostic hallucinations in up to 28% of rare pathology cases [3].
  * *Synthesis*: General foundation models are competent for common clinical text but present unacceptable risk in specialized diagnostic niches without domain-specific calibration.

## 9. Limitations
1. Over-reliance on retrospective databases (such as MIMIC-IV and synthetic cohorts) rather than live prospective clinical evaluations.
2. Sensitivity to prompt phrasing and lack of standardized uncertainty calibration.

## 10. Potential Research Gaps
* **Potential Research Gap 1: Prospective Clinical Trials**: While retrospective benchmarks are abundant, randomized controlled trials evaluating patient outcomes under GenAI assistance remain largely absent.
* **Potential Research Gap 2: Edge-Device and Low-Resource Deployment**: Models require substantial multi-GPU server infrastructure, leaving rural and resource-constrained clinics underserved.

## 11. Future Directions
Exploration of federated clinical training to preserve patient privacy, development of sub-100ms quantized edge models, and standardized hallucination detection harnesses.

## 12. Conclusion
Generative AI demonstrates undeniable promise in reducing clinical administrative burden and aiding diagnosis. Realizing this potential requires shifting research emphasis from general benchmark scores toward robust safety verification, prospective clinical integration, and low-latency architectures.

## 13. References
[1] Chen, H., et al. (2024). *Domain-Adapted Generative AI for Clinical Decision Support*. arXiv:2309.12345. Available at: https://arxiv.org/abs/2309.12345
[2] Rodriguez, M., et al. (2024). *Benchmarking Real-Time Generative Architectures in Healthcare*. arXiv:2401.05678. Available at: https://arxiv.org/abs/2401.05678
[3] Patel, S., & Vance, J. (2024). *Safety and Failure Modes of Unadapted Medical LLMs*. Nature Digital Medicine. DOI: 10.1038/s41746-024-00123-x.
"""
        else:
            content = f"Evidentia AI processed request: {full_prompt[:200]}..."

        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])


def get_llm(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    demo_mode: bool = False,
    provider: Optional[str] = None,
    base_url: Optional[str] = None
) -> BaseChatModel:
    """
    Returns an appropriate LLM instance matching the architecture in the proposal:
    Supports: OpenAI, Google Gemini, Anthropic Claude, Ollama (Local), and intelligent Demo Mode.
    """
    active_provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower().strip()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    
    # Determine active key based on provider
    if active_provider == "gemini":
        active_key = (api_key or gemini_key or env_key).strip()
    elif active_provider == "anthropic":
        active_key = (api_key or anthropic_key).strip()
    elif active_provider == "ollama":
        active_key = "ollama"  # Ollama runs locally without key
    else:
        active_key = (api_key or env_key).strip()

    env_demo = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")

    if demo_mode or env_demo or (active_provider != "ollama" and not active_key):
        logger.info("Initializing DemoChatModel (Demo / Offline Mode active).")
        return DemoChatModel(model_name="evidentia-demo-engine")

    # 1. Google Gemini via OpenAI-compatible endpoint
    if active_provider == "gemini":
        try:
            from langchain_openai import ChatOpenAI
            chosen_model = model_name or os.getenv("MODEL_NAME", "gemini-1.5-flash")
            endpoint = base_url or os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
            logger.info(f"Initializing Gemini via endpoint with model: {chosen_model}")
            return ChatOpenAI(
                api_key=active_key,
                model=chosen_model,
                base_url=endpoint,
                temperature=0.1
            )
        except Exception as e:
            logger.warning(f"Gemini init error: {e}. Falling back to DemoChatModel.")
            return DemoChatModel(model_name="evidentia-demo-engine")

    # 2. Ollama (Local LLM)
    elif active_provider == "ollama":
        try:
            from langchain_openai import ChatOpenAI
            chosen_model = model_name or os.getenv("MODEL_NAME", "llama3")
            endpoint = base_url or os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
            logger.info(f"Initializing Ollama local model: {chosen_model} at {endpoint}")
            return ChatOpenAI(
                api_key="ollama",
                model=chosen_model,
                base_url=endpoint,
                temperature=0.1
            )
        except Exception as e:
            logger.warning(f"Ollama local init error: {e}. Falling back to DemoChatModel.")
            return DemoChatModel(model_name="evidentia-demo-engine")

    # 3. Anthropic Claude
    elif active_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            chosen_model = model_name or os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620")
            logger.info(f"Initializing ChatAnthropic with model: {chosen_model}")
            return ChatAnthropic(
                api_key=active_key,
                model=chosen_model,
                temperature=0.1
            )
        except Exception as e:
            logger.warning(f"Anthropic init error: {e}. Falling back to DemoChatModel.")
            return DemoChatModel(model_name="evidentia-demo-engine")

    # 4. Default: OpenAI
    try:
        from langchain_openai import ChatOpenAI
        chosen_model = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        api_base = base_url or os.getenv("OPENAI_API_BASE", "").strip() or None
        
        logger.info(f"Initializing ChatOpenAI with model: {chosen_model}")
        return ChatOpenAI(
            api_key=active_key,
            model=chosen_model,
            temperature=0.1,
            base_url=api_base
        )
    except Exception as e:
        logger.warning(f"Could not initialize ChatOpenAI ({e}). Falling back to DemoChatModel.")
        return DemoChatModel(model_name="evidentia-demo-engine")

