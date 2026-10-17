"""
Tests for document splitter and chunking logic.
"""

import pytest
from langchain_core.documents import Document
from rag_assistant.splitter import DocumentSplitter


def test_splitter_invalid_parameters():
    with pytest.raises(ValueError):
        DocumentSplitter(chunk_size=40, chunk_overlap=10)

    with pytest.raises(ValueError):
        DocumentSplitter(chunk_size=500, chunk_overlap=600)

    with pytest.raises(ValueError):
        DocumentSplitter(chunk_size=500, chunk_overlap=-10)


def test_splitter_chunking_and_metadata():
    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)

    sample_text = (
        "In the High Court of Judicature. Criminal Case No. CC/482/2024. "
        "The defendant Vikram Chowdary was charged under sections 420, 468, and 471 of the IPC. "
        "The prosecution presented three witnesses including the forensic handwriting expert. "
        "The trial court issued notice on the interlocutory application."
    )

    doc = Document(
        page_content=sample_text,
        metadata={"filename": "chowdary.txt", "source": "data/chowdary.txt", "page": 1}
    )

    chunks = splitter.split_documents([doc])
    assert len(chunks) > 1

    # Check metadata enrichment
    for idx, c in enumerate(chunks):
        assert c.metadata["chunk_id"] == idx
        assert c.metadata["filename"] == "chowdary.txt"
        assert c.metadata["page"] == 1
        assert "chunk_char_count" in c.metadata
        assert "chunk_preview" in c.metadata
        assert len(c.page_content) <= 120  # Respects chunk size with margin


def test_splitter_empty_documents():
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=20)
    assert splitter.split_documents([]) == []
    stats = splitter.get_chunking_stats([])
    assert stats["total_chunks"] == 0
    assert stats["avg_char_length"] == 0.0
