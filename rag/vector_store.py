import os
import shutil
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from rag.embeddings import get_embedding_function
from utils.logging import logger

DEFAULT_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

class VectorStoreManager:
    """
    Manages vector storage and retrieval using ChromaDB.
    Designed with a swappable interface to support FAISS/Qdrant in future updates.
    """
    def __init__(
        self,
        persist_directory: str = DEFAULT_CHROMA_DIR,
        collection_name: str = "evidentia_research",
        api_key: Optional[str] = None,
        demo_mode: bool = False
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_fn = get_embedding_function(api_key=api_key, demo_mode=demo_mode)
        self.client = None
        self.collection = None
        self._init_store()

    def _init_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Evidentia AI Academic Research Embeddings"}
            )
            logger.info(f"ChromaDB initialized at '{self.persist_directory}' collection '{self.collection_name}'.")
        except Exception as e:
            logger.warning(f"Persistent ChromaDB init failed ({e}). Falling back to ephemeral in-memory client.")
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False, is_persistent=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

    def add_chunks(self, chunks: List[Dict[str, Any]], session_id: str = "default") -> int:
        """
        Embeds and stores document chunks in the vector collection.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            cid = chunk.get("chunk_id", f"{session_id}_chunk_{idx}")
            text = chunk.get("content", "").strip()
            if not text:
                continue

            meta = chunk.get("metadata", {}).copy()
            meta["session_id"] = session_id
            # Ensure metadata values are strings, ints, floats, or bools for ChromaDB
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)

            ids.append(f"{session_id}_{cid}_{idx}")
            documents.append(text)
            metadatas.append(clean_meta)

        if not documents:
            return 0

        # Generate embeddings
        embeddings = self.embedding_fn.embed_documents(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Stored {len(documents)} chunks in ChromaDB for session {session_id}.")
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search for relevant evidence chunks.
        """
        if not query.strip() or self.collection.count() == 0:
            return []

        query_embedding = self.embedding_fn.embed_query(query)
        where_filter = {"session_id": session_id} if session_id else None

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.warning(f"ChromaDB search query failed with filter ({e}). Retrying without filter.")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )

        retrieved: List[Dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            retrieved.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "relevance_score": max(0.0, 1.0 - dist) if dist is not None else 1.0
            })

        logger.info(f"Retrieved {len(retrieved)} relevant chunks for query: '{query[:40]}...'")
        return retrieved

    def count(self) -> int:
        """Returns total documents stored in collection."""
        return self.collection.count() if self.collection else 0
