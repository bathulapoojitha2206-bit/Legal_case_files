"""
Tests for FAISS vector store creation, serialization, deserialization, and querying.
"""

import pytest
from langchain_core.documents import Document
from rag_assistant.embeddings import EmbeddingManager
from rag_assistant.vectorstore import FAISSVectorStoreManager, IndexNotFoundError


@pytest.fixture(scope="module")
def shared_embeddings():
    return EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu").get_embeddings()


def test_vectorstore_lifecycle(tmp_path, shared_embeddings):
    index_dir = tmp_path / "test_faiss_index"

    manager = FAISSVectorStoreManager(index_dir=index_dir, embeddings=shared_embeddings)
    assert manager.index_exists() is False

    # 1. Build index from sample chunks
    docs = [
        Document(
            page_content="Vikram Chowdary was charged under section 420 for cheating in Hyderabad.",
            metadata={"filename": "chowdary.pdf", "page": 1, "chunk_id": 0}
        ),
        Document(
            page_content="BlueWave Constructions filed a commercial dispute for unpaid milestone dues.",
            metadata={"filename": "bluewave.pdf", "page": 2, "chunk_id": 1}
        ),
        Document(
            page_content="Lakshmi Devi filed a title suit for 2.5 acres of land in Shankarpally.",
            metadata={"filename": "lakshmidevi.pdf", "page": 1, "chunk_id": 2}
        ),
    ]

    manager.build_index(docs)
    assert manager.vector_store is not None

    stats = manager.get_index_stats()
    assert stats["total_vectors"] == 3
    assert stats["dimension"] == 384

    # 2. Persist index
    saved_path = manager.save_index()
    assert manager.index_exists() is True
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()

    # 3. Reload from disk with fresh manager
    reloaded_manager = FAISSVectorStoreManager(index_dir=index_dir, embeddings=shared_embeddings)
    reloaded_manager.load_index()
    assert reloaded_manager.vector_store is not None
    assert reloaded_manager.get_index_stats()["total_vectors"] == 3

    # 4. Perform similarity search
    results = reloaded_manager.similarity_search_with_score("Who was charged under section 420?", k=1)
    assert len(results) == 1
    matched_doc, score = results[0]
    assert matched_doc.metadata["filename"] == "chowdary.pdf"
    assert "420" in matched_doc.page_content


def test_vectorstore_missing_index_raises(tmp_path, shared_embeddings):
    empty_dir = tmp_path / "non_existent_index"
    manager = FAISSVectorStoreManager(index_dir=empty_dir, embeddings=shared_embeddings)

    with pytest.raises(IndexNotFoundError):
        manager.load_index()

    with pytest.raises(IndexNotFoundError):
        manager.similarity_search_with_score("any query", k=1)


def test_vectorstore_empty_document_list(tmp_path, shared_embeddings):
    manager = FAISSVectorStoreManager(index_dir=tmp_path, embeddings=shared_embeddings)
    with pytest.raises(ValueError):
        manager.build_index([])
