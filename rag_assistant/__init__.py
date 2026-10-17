"""
Legal Case Files - RAG Knowledge Assistant
Built with LangChain, Hugging Face Embeddings, FAISS, and Ollama.
"""

from rag_assistant.config import RAGConfig
from rag_assistant.loader import DocumentLoader, LoadedDocument
from rag_assistant.splitter import DocumentSplitter
from rag_assistant.embeddings import EmbeddingManager
from rag_assistant.vectorstore import FAISSVectorStoreManager
from rag_assistant.retriever import LegalRetriever
from rag_assistant.generator import LegalResponseGenerator
from rag_assistant.pipeline import LegalRAGPipeline

__all__ = [
    "RAGConfig",
    "DocumentLoader",
    "LoadedDocument",
    "DocumentSplitter",
    "EmbeddingManager",
    "FAISSVectorStoreManager",
    "LegalRetriever",
    "LegalResponseGenerator",
    "LegalRAGPipeline",
]

__version__ = "1.0.0"
