"""
End-to-end RAG Pipeline orchestrator for Legal Case Files.
Coordinates ingestion, splitting, FAISS indexing, retrieval, and Ollama generation.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document

from rag_assistant.config import RAGConfig
from rag_assistant.embeddings import EmbeddingManager
from rag_assistant.generator import (
    GenerationResult,
    LegalResponseGenerator,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from rag_assistant.loader import DocumentLoader, LoadedDocument
from rag_assistant.retriever import LegalRetriever, RetrievedPassage
from rag_assistant.splitter import DocumentSplitter
from rag_assistant.vectorstore import FAISSVectorStoreManager, IndexNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    """Summary of the document ingestion and indexing run."""
    total_files_scanned: int
    files_indexed: List[str]
    files_skipped_or_failed: List[Dict[str, str]]
    total_chunks: int
    avg_chunk_length: float
    index_saved_path: str
    duration_seconds: float


@dataclass
class PipelineResult:
    """Full structured response from a question-answering query."""
    question: str
    answer: str
    passages: List[RetrievedPassage]
    sources: List[Dict[str, Any]]
    model_used: str
    provider: str
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    status: str = "success"
    error_message: Optional[str] = None


class LegalRAGPipeline:
    """
    Main entry point for the Legal RAG Knowledge Assistant.
    Provides complete lifecycle methods: ingestion, indexing, retrieval, and QA.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.config.validate()

        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device
        )
        self.vector_store_manager = FAISSVectorStoreManager(
            index_dir=self.config.faiss_index_dir,
            embeddings=self.embedding_manager.get_embeddings()
        )
        self.retriever = LegalRetriever(
            vector_store_manager=self.vector_store_manager,
            k=self.config.retriever_k,
            use_reranker=self.config.use_reranker,
            reranker_model=self.config.reranker_model
        )
        self.generator = LegalResponseGenerator(
            provider=self.config.llm_provider,
            model_name=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.ollama_temperature,
            request_timeout=self.config.ollama_request_timeout,
            openai_api_key=self.config.openai_api_key,
            openai_model=self.config.openai_model,
            groq_api_key=self.config.groq_api_key,
            groq_model=self.config.groq_model
        )

    def ingest_directory(
        self,
        directory_path: Union[str, Path] = "./data/sample_documents",
        save_index: bool = True
    ) -> IngestionReport:
        """
        Ingest all supported documents in a directory and build the FAISS index.

        Args:
            directory_path: Path to document directory.
            save_index: Whether to persist the generated index to disk.

        Returns:
            IngestionReport with detailed stats.
        """
        start_time = time.time()
        dir_p = Path(directory_path).resolve()

        all_docs, load_results = self.loader.load_directory(dir_p)

        indexed_files = []
        failed_files = []

        for res in load_results:
            if res.success and res.documents:
                indexed_files.append(res.filename)
            else:
                reason = res.error_message or res.warning_message or "Unknown failure"
                failed_files.append({"filename": res.filename, "reason": reason})

        if not all_docs:
            elapsed = time.time() - start_time
            return IngestionReport(
                total_files_scanned=len(load_results),
                files_indexed=[],
                files_skipped_or_failed=failed_files,
                total_chunks=0,
                avg_chunk_length=0.0,
                index_saved_path="",
                duration_seconds=round(elapsed, 2)
            )

        # Split into chunks
        chunks = self.splitter.split_documents(all_docs)
        stats = self.splitter.get_chunking_stats(chunks)

        # Build FAISS vector store
        self.vector_store_manager.build_index(chunks)

        saved_path = ""
        if save_index:
            saved_path = str(self.vector_store_manager.save_index())

        elapsed = time.time() - start_time
        return IngestionReport(
            total_files_scanned=len(load_results),
            files_indexed=indexed_files,
            files_skipped_or_failed=failed_files,
            total_chunks=stats["total_chunks"],
            avg_chunk_length=stats["avg_char_length"],
            index_saved_path=saved_path,
            duration_seconds=round(elapsed, 2)
        )

    def ask(self, question: str, top_k: Optional[int] = None) -> PipelineResult:
        """
        Process a question through retrieval and grounded generation.

        Args:
            question: The user query string.
            top_k: Optional top-k documents override.

        Returns:
            PipelineResult containing answer, citations, timing, and status.
        """
        start_total = time.time()
        q_clean = question.strip()

        # Handle empty question gracefully
        if not q_clean:
            return PipelineResult(
                question="",
                answer="Please enter a valid legal question to search the case files.",
                passages=[],
                sources=[],
                model_used=self.config.ollama_model,
                provider=self.config.llm_provider,
                retrieval_time_sec=0.0,
                generation_time_sec=0.0,
                total_time_sec=0.0,
                status="warning",
                error_message="Empty question string provided."
            )

        # Verify index availability
        if not self.vector_store_manager.index_exists() and self.vector_store_manager.vector_store is None:
            return PipelineResult(
                question=q_clean,
                answer=(
                    "No legal documents have been indexed yet. "
                    "Please run document ingestion to create the FAISS index before asking questions."
                ),
                passages=[],
                sources=[],
                model_used=self.config.ollama_model,
                provider=self.config.llm_provider,
                retrieval_time_sec=0.0,
                generation_time_sec=0.0,
                total_time_sec=0.0,
                status="warning",
                error_message="Missing FAISS vector store index."
            )

        # 1. Retrieval Phase
        t0 = time.time()
        try:
            passages = self.retriever.retrieve(q_clean, top_k=top_k)
            retrieval_time = time.time() - t0
        except IndexNotFoundError as err:
            return PipelineResult(
                question=q_clean,
                answer="FAISS index is missing. Please ingest case files first.",
                passages=[],
                sources=[],
                model_used=self.config.ollama_model,
                provider=self.config.llm_provider,
                retrieval_time_sec=0.0,
                generation_time_sec=0.0,
                total_time_sec=time.time() - start_total,
                status="error",
                error_message=str(err)
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            return PipelineResult(
                question=q_clean,
                answer=f"Error occurred during passage retrieval: {exc}",
                passages=[],
                sources=[],
                model_used=self.config.ollama_model,
                provider=self.config.llm_provider,
                retrieval_time_sec=time.time() - t0,
                generation_time_sec=0.0,
                total_time_sec=time.time() - start_total,
                status="error",
                error_message=str(exc)
            )

        # 2. Generation Phase
        t1 = time.time()
        try:
            gen_result = self.generator.generate(q_clean, passages)
            generation_time = time.time() - t1
            total_time = time.time() - start_total

            sources = [p.to_dict() for p in passages]

            return PipelineResult(
                question=q_clean,
                answer=gen_result.answer,
                passages=passages,
                sources=sources,
                model_used=gen_result.model,
                provider=gen_result.provider,
                retrieval_time_sec=round(retrieval_time, 3),
                generation_time_sec=round(generation_time, 3),
                total_time_sec=round(total_time, 3),
                status="success"
            )

        except OllamaConnectionError as err:
            return PipelineResult(
                question=q_clean,
                answer=(
                    f"⚠️ Ollama is not accessible: {err}\n\n"
                    "**Troubleshooting Steps:**\n"
                    "1. Open terminal and run: `ollama serve`\n"
                    "2. Check that Ollama is running at `http://localhost:11434`\n"
                    "3. Verify your model is installed: `ollama list`"
                ),
                passages=passages,
                sources=[p.to_dict() for p in passages],
                model_used=self.config.ollama_model,
                provider="ollama",
                retrieval_time_sec=round(retrieval_time, 3),
                generation_time_sec=round(time.time() - t1, 3),
                total_time_sec=round(time.time() - start_total, 3),
                status="error",
                error_message=str(err)
            )

        except OllamaModelNotFoundError as err:
            return PipelineResult(
                question=q_clean,
                answer=(
                    f"⚠️ Missing Ollama Model: {err}\n\n"
                    f"Run `ollama pull {self.config.ollama_model}` to download the model weights."
                ),
                passages=passages,
                sources=[p.to_dict() for p in passages],
                model_used=self.config.ollama_model,
                provider="ollama",
                retrieval_time_sec=round(retrieval_time, 3),
                generation_time_sec=round(time.time() - t1, 3),
                total_time_sec=round(time.time() - start_total, 3),
                status="error",
                error_message=str(err)
            )

        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            return PipelineResult(
                question=q_clean,
                answer=f"Error generating answer: {exc}",
                passages=passages,
                sources=[p.to_dict() for p in passages],
                model_used=self.config.ollama_model,
                provider=self.config.llm_provider,
                retrieval_time_sec=round(retrieval_time, 3),
                generation_time_sec=round(time.time() - t1, 3),
                total_time_sec=round(time.time() - start_total, 3),
                status="error",
                error_message=str(exc)
            )
