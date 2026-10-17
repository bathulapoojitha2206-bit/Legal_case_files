"""
End-to-end integration tests for LegalRAGPipeline.
"""

import pytest
from rag_assistant.config import RAGConfig
from rag_assistant.pipeline import LegalRAGPipeline


def test_pipeline_empty_question(tmp_path):
    config = RAGConfig(
        faiss_index_dir=str(tmp_path / "index"),
        llm_provider="mock"
    )
    pipeline = LegalRAGPipeline(config=config)
    res = pipeline.ask("")
    assert res.status == "warning"
    assert "valid legal question" in res.answer


def test_pipeline_missing_index(tmp_path):
    config = RAGConfig(
        faiss_index_dir=str(tmp_path / "missing_index"),
        llm_provider="mock"
    )
    pipeline = LegalRAGPipeline(config=config)
    res = pipeline.ask("What is Vikram Chowdary charged with?")
    assert res.status == "warning"
    assert "No legal documents have been indexed" in res.answer


def test_pipeline_end_to_end_mock(tmp_path):
    # 1. Create temporary case documents
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    case1 = docs_dir / "01_criminal_case.txt"
    case1.write_text(
        "State of Telangana vs. Vikram Chowdary. Criminal Case CC/482/2024. "
        "Charged under IPC Sections 420, 468, and 471 for bank loan forgery.",
        encoding="utf-8"
    )

    case2 = docs_dir / "02_commercial_case.txt"
    case2.write_text(
        "BlueWave Constructions vs. Orion Infra. CS(COMM)/112/2024. "
        "Clause 18 mandates arbitration in Hyderabad under Arbitration Act 1996.",
        encoding="utf-8"
    )

    index_dir = tmp_path / "index"
    config = RAGConfig(
        faiss_index_dir=str(index_dir),
        llm_provider="mock",
        retriever_k=2
    )
    pipeline = LegalRAGPipeline(config=config)

    # Ingest documents
    report = pipeline.ingest_directory(docs_dir, save_index=True)
    assert report.total_chunks >= 2
    assert len(report.files_indexed) == 2
    assert pipeline.vector_store_manager.index_exists() is True

    # Query pipeline
    result = pipeline.ask("What penal sections is Vikram Chowdary charged with?")
    assert result.status == "success"
    assert len(result.passages) > 0
    assert result.sources[0]["filename"] == "01_criminal_case.txt"
    assert result.total_time_sec > 0
