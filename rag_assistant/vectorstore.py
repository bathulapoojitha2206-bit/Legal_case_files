"""
FAISS Vector Store Manager for indexing, persistence, and vector search.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class IndexNotFoundError(Exception):
    """Raised when an operation requires an index that has not been created or saved."""
    pass


class FAISSVectorStoreManager:
    """Manages creation, serialization, deserialization, and query of FAISS vector stores."""

    def __init__(
        self,
        index_dir: Union[str, Path] = "./faiss_index",
        embeddings: Optional[Embeddings] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu"
    ):
        self.index_dir = Path(index_dir).resolve()
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = EmbeddingManager(
                model_name=embedding_model_name,
                device=embedding_device
            ).get_embeddings()

        self.vector_store: Optional[FAISS] = None

    def index_exists(self, custom_dir: Optional[Union[str, Path]] = None) -> bool:
        """Check if a valid FAISS index exists on disk."""
        target_dir = Path(custom_dir).resolve() if custom_dir else self.index_dir
        faiss_file = target_dir / "index.faiss"
        pkl_file = target_dir / "index.pkl"
        return target_dir.exists() and faiss_file.exists() and pkl_file.exists()

    def build_index(self, documents: List[Document]) -> FAISS:
        """
        Build a FAISS vector store from a collection of split documents.

        Args:
            documents: List of chunked Document objects.

        Returns:
            Instantiated FAISS vector store.
        """
        if not documents:
            raise ValueError("Cannot build FAISS index from empty document list.")

        logger.info("Building FAISS index with %d chunks...", len(documents))
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return self.vector_store

    def save_index(self, custom_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Persist the FAISS index and docstore pickle to disk.

        Args:
            custom_dir: Optional override path for storage.

        Returns:
            Resolved Path to saved index directory.
        """
        if self.vector_store is None:
            raise IndexNotFoundError("No in-memory FAISS vector store to save. Build or load one first.")

        target_dir = Path(custom_dir).resolve() if custom_dir else self.index_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving FAISS index to: %s", target_dir)
        self.vector_store.save_local(str(target_dir))
        return target_dir

    def load_index(self, custom_dir: Optional[Union[str, Path]] = None) -> FAISS:
        """
        Load an existing FAISS index from disk.

        Args:
            custom_dir: Optional path to index folder.

        Returns:
            Loaded FAISS vector store.
        """
        target_dir = Path(custom_dir).resolve() if custom_dir else self.index_dir

        if not self.index_exists(target_dir):
            raise IndexNotFoundError(
                f"FAISS index not found at '{target_dir}'. "
                "Please index your legal documents first before querying."
            )

        logger.info("Loading FAISS index from: %s", target_dir)
        # allow_dangerous_deserialization=True is required by LangChain to load the local docstore pickle
        self.vector_store = FAISS.load_local(
            folder_path=str(target_dir),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """Add additional documents to the existing in-memory index."""
        if not documents:
            return
        if self.vector_store is None:
            self.build_index(documents)
        else:
            self.vector_store.add_documents(documents)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents alongside similarity scores.

        Args:
            query: The search query string.
            k: Top-k documents to retrieve.

        Returns:
            List of (Document, score) tuples.
        """
        if self.vector_store is None:
            if self.index_exists():
                self.load_index()
            else:
                raise IndexNotFoundError(
                    "No FAISS index available. Please build or load an index first."
                )

        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_index_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the current index."""
        if self.vector_store is None:
            if self.index_exists():
                try:
                    self.load_index()
                except Exception:
                    return {
                        "status": "unloaded",
                        "exists_on_disk": True,
                        "path": str(self.index_dir),
                        "total_vectors": 0
                    }
            else:
                return {
                    "status": "missing",
                    "exists_on_disk": False,
                    "path": str(self.index_dir),
                    "total_vectors": 0
                }

        num_vectors = self.vector_store.index.ntotal if hasattr(self.vector_store, "index") else 0
        dim = self.vector_store.index.d if hasattr(self.vector_store, "index") else 0

        return {
            "status": "ready",
            "exists_on_disk": self.index_exists(),
            "path": str(self.index_dir),
            "total_vectors": num_vectors,
            "dimension": dim,
        }
