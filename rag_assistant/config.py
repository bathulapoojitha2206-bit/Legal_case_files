"""
Configuration module for the Legal Case Files RAG Knowledge Assistant.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class RAGConfig:
    """Central configuration class for the RAG pipeline."""

    # LLM Settings
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    ollama_temperature: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TEMPERATURE", "0.1")))
    ollama_request_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "60.0")))

    # Hugging Face Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))

    # Chunking & Splitting
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "600")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))

    # FAISS Vector Store
    faiss_index_dir: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_DIR", "./faiss_index"))

    # Retrieval
    retriever_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_K", "4")))
    use_reranker: bool = field(
        default_factory=lambda: os.getenv("USE_RERANKER", "false").lower() in ("true", "1", "yes")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # Optional Cloud LLM Fallbacks (for deployment on hosts without local Ollama)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.2-3b-preview"))

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size <= 50:
            raise ValueError(f"chunk_size must be > 50, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})"
            )
        if self.retriever_k <= 0:
            raise ValueError(f"retriever_k must be > 0, got {self.retriever_k}")

    @property
    def index_path(self) -> Path:
        """Return the resolved Path to the FAISS index directory."""
        return Path(self.faiss_index_dir).resolve()
