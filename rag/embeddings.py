import os
import hashlib
import numpy as np
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from utils.logging import logger

class DeterministicDemoEmbeddings(Embeddings):
    """
    Fast, deterministic embeddings provider for demo mode, testing, and offline execution.
    Generates 384-dimensional normalized vectors based on token hashes and n-grams.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec.tolist()
            
        for token in tokens:
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[idx] += sign

        # Add character bi-grams for semantic substring matching
        for i in range(len(text) - 1):
            bg = text[i:i+2].lower()
            h = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            vec[idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedding_function(api_key: Optional[str] = None, demo_mode: bool = False) -> Embeddings:
    """
    Returns OpenAI embeddings if API key is provided and demo_mode is False,
    otherwise returns DeterministicDemoEmbeddings.
    """
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    active_key = (api_key or env_key).strip()
    env_demo = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")

    if demo_mode or env_demo or not active_key:
        logger.info("Using DeterministicDemoEmbeddings (Demo/Offline Mode).")
        return DeterministicDemoEmbeddings()

    try:
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        logger.info(f"Using OpenAIEmbeddings with model: {model}")
        return OpenAIEmbeddings(api_key=active_key, model=model)
    except Exception as e:
        logger.warning(f"Could not load OpenAIEmbeddings ({e}). Falling back to DeterministicDemoEmbeddings.")
        return DeterministicDemoEmbeddings()
