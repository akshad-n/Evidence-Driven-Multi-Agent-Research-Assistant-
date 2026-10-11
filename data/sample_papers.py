from typing import List, Dict, Any

DEMO_PAPERS: List[Dict[str, Any]] = [
    {
        "title": "A Survey of Generative AI in Healthcare: New Prospects and Challenges",
        "authors": ["Chun-Kai Wang", "Chun-Yen Chen", "Hao-Chun Yang", "Yen-Chun Lin"],
        "year": 2024,
        "abstract": "Generative Artificial Intelligence (GenAI) is reshaping healthcare through automated clinical documentation, medical imaging synthesis, and diagnostic decision support. We synthesize 120+ recent studies, showing clinical documentation time reductions of up to 40%. However, critical challenges regarding hallucination rates, privacy preservation under HIPAA, and model explainability persist.",
        "url": "https://arxiv.org/abs/2403.01234",
        "doi": "10.48550/arXiv.2403.01234",
        "source": "arxiv"
    },
    {
        "title": "Large Language Models Encode Clinical Knowledge: Med-PaLM Benchmarks",
        "authors": ["Karan Singhal", "Shekoofeh Azizi", "Tao Tu", "S. Sara Mahdavi", "Jason Wei"],
        "year": 2023,
        "abstract": "We evaluate clinical reasoning capabilities using MedQA, MedMCQA, and PubMedQA. Domain-tuned medical foundation models achieved 85.4% accuracy on USMLE-style questions. However, expert clinician evaluations revealed subtle omissions and highlighted the absence of real-world prospective bedside validation.",
        "url": "https://doi.org/10.1038/s41586-023-06291-2",
        "doi": "10.1038/s41586-023-06291-2",
        "source": "nature"
    },
    {
        "title": "Evaluation and Mitigation of Hallucinations in Clinical Generative AI",
        "authors": ["Hao Chen", "Michael Rodriguez", "Emily Vance"],
        "year": 2024,
        "abstract": "Evaluating factual accuracy in clinician-facing EHR summarization. Unadapted general foundation models produce up to 24% inaccurate clinical assertions on complex pathology. In contrast, instruction tuning on verified EHR trajectories combined with retrieval grounding reduces hallucination rates to below 4%.",
        "url": "https://arxiv.org/abs/2401.09876",
        "doi": "10.48550/arXiv.2401.09876",
        "source": "arxiv"
    },
    {
        "title": "Real-Time Inference Latency Benchmarks for Intraoperative AI Assistance",
        "authors": ["Marcus Wei", "David Lee", "Sarah Connor"],
        "year": 2024,
        "abstract": "Evaluating transformer inference latency in surgical and emergency settings. Experimental benchmarks demonstrate that response latencies exceeding 2.0 seconds cause significant operational bottlenecks in intraoperative guidance. Quantized architectures achieve 180ms latency but require edge-GPU infrastructure not available in low-resource clinics.",
        "url": "https://doi.org/10.1109/TBME.2024.12345",
        "doi": "10.1109/TBME.2024.12345",
        "source": "crossref"
    },
    {
        "title": "Ethical and Regulatory Constraints in Deploying Medical Foundation Models",
        "authors": ["Amara Okafor", "Julian Sterling", "Elena Rostova"],
        "year": 2024,
        "abstract": "Analyzing the regulatory landscape for FDA software-as-a-medical-device (SaMD) clearances of generative models. Highlights the legal liability of probabilistic outputs, data bias against underrepresented demographic cohorts, and the lack of prospective clinical trial frameworks.",
        "url": "https://doi.org/10.1016/j.lanepe.2024.10089",
        "doi": "10.1016/j.lanepe.2024.10089",
        "source": "semanticscholar"
    }
]

def get_demo_papers(query: str = "", limit: int = 5) -> List[Dict[str, Any]]:
    """Returns local demo papers matching or filtered."""
    return DEMO_PAPERS[:limit]
