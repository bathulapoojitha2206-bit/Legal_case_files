"""
Tests for response generator, grounding checks, and prompt assembly.
"""

from langchain_core.documents import Document
from rag_assistant.generator import LegalResponseGenerator
from rag_assistant.retriever import RetrievedPassage


def test_format_context_with_passages():
    generator = LegalResponseGenerator(provider="mock")

    passages = [
        RetrievedPassage(
            rank=1,
            content="Vikram Chowdary was charged under IPC 420 for cheating.",
            filename="chowdary.pdf",
            source="data/chowdary.pdf",
            page=1,
            chunk_id=0,
            score=0.88,
            raw_document=Document(page_content="...")
        )
    ]

    formatted = generator.format_context(passages)
    assert "chowdary.pdf" in formatted
    assert "Page 1" in formatted
    assert "Chunk #0" in formatted
    assert "IPC 420" in formatted


def test_generate_empty_question():
    generator = LegalResponseGenerator(provider="mock")
    res = generator.generate("", [])
    assert "valid" in res.answer.lower()
    assert res.passages == []


def test_generate_no_passages_grounding_rejection():
    # If no passages are retrieved, the generator must state information is unavailable
    generator = LegalResponseGenerator(provider="mock")
    res = generator.generate("What is the admiralty law ruling on vessel salvage?", [])
    assert "not available" in res.answer.lower()


def test_mock_generation():
    generator = LegalResponseGenerator(provider="mock")
    passages = [
        RetrievedPassage(
            rank=1,
            content="Clause 18 mandates arbitration in Hyderabad.",
            filename="contract.pdf",
            source="data/contract.pdf",
            page=3,
            chunk_id=5,
            score=0.92,
            raw_document=Document(page_content="...")
        )
    ]
    res = generator.generate("What does Clause 18 state?", passages)
    assert "contract.pdf" in res.answer
    assert res.provider == "mock"
    assert len(res.passages) == 1
