"""
Retrieval module with similarity search and optional Cross-Encoder reranking.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from rag_assistant.vectorstore import FAISSVectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    """Structured representation of a retrieved document passage."""
    rank: int
    content: str
    filename: str
    source: str
    page: Optional[int]
    chunk_id: Optional[int]
    score: float
    raw_document: Document

    def to_dict(self) -> Dict[str, Any]:
        """Convert passage to JSON-serializable dictionary."""
        return {
            "rank": self.rank,
            "filename": self.filename,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "content": self.content,
            "source": self.source,
        }


class LegalRetriever:
    """
    Retrieves and ranks relevant passages for legal questions.
    Supports standard FAISS similarity search and Cross-Encoder neural reranking.
    """

    _reranker_instance = None
    _reranker_model_name = None

    def __init__(
        self,
        vector_store_manager: FAISSVectorStoreManager,
        k: int = 4,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.vector_store_manager = vector_store_manager
        self.k = max(1, k)
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model

    def _get_reranker(self):
        """Lazy loader for sentence-transformers CrossEncoder."""
        if not self.use_reranker:
            return None

        if (
            LegalRetriever._reranker_instance is not None
            and LegalRetriever._reranker_model_name == self.reranker_model
        ):
            return LegalRetriever._reranker_instance

        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder reranker: %s", self.reranker_model)
            LegalRetriever._reranker_instance = CrossEncoder(self.reranker_model)
            LegalRetriever._reranker_model_name = self.reranker_model
            return LegalRetriever._reranker_instance
        except Exception as exc:
            logger.warning("Failed to load CrossEncoder reranker '%s': %s. Falling back to vector similarity.", self.reranker_model, exc)
            self.use_reranker = False
            return None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedPassage]:
        """
        Retrieve relevant passages for a query with optional reranking.

        Args:
            query: The user's legal question.
            top_k: Optional override for k.

        Returns:
            List of RetrievedPassage objects ranked by relevance.
        """
        k = top_k if top_k is not None and top_k > 0 else self.k
        query_clean = query.strip()
        if not query_clean:
            return []

        # If reranking is enabled, fetch a wider pool of candidates first
        initial_k = max(k * 2, 8) if self.use_reranker else k

        results = self.vector_store_manager.similarity_search_with_score(
            query=query_clean,
            k=initial_k
        )

        if not results:
            return []

        # Check if reranking can be applied
        reranker = self._get_reranker()
        if reranker is not None and len(results) > 1:
            try:
                pairs = [(query_clean, doc.page_content) for doc, _ in results]
                scores = reranker.predict(pairs)

                # Pair with documents and sort descending
                scored_docs = list(zip(scores, [doc for doc, _ in results]))
                scored_docs.sort(key=lambda x: x[0], reverse=True)

                # Take top k
                final_results = scored_docs[:k]
                passages = []
                for rank, (score, doc) in enumerate(final_results, start=1):
                    meta = doc.metadata or {}
                    passages.append(
                        RetrievedPassage(
                            rank=rank,
                            content=doc.page_content,
                            filename=meta.get("filename", "Unknown"),
                            source=meta.get("source", "Unknown"),
                            page=meta.get("page"),
                            chunk_id=meta.get("chunk_id"),
                            score=float(score),
                            raw_document=doc,
                        )
                    )
                return passages
            except Exception as exc:
                logger.warning("Reranking failed (%s). Falling back to direct vector scores.", exc)

        # Standard vector similarity ranking
        passages = []
        for rank, (doc, raw_distance) in enumerate(results[:k], start=1):
            meta = doc.metadata or {}
            # FAISS L2 or cosine: normalize score for readability
            score = 1.0 / (1.0 + float(raw_distance))
            passages.append(
                RetrievedPassage(
                    rank=rank,
                    content=doc.page_content,
                    filename=meta.get("filename", "Unknown"),
                    source=meta.get("source", "Unknown"),
                    page=meta.get("page"),
                    chunk_id=meta.get("chunk_id"),
                    score=round(score, 4),
                    raw_document=doc,
                )
            )

        return passages
