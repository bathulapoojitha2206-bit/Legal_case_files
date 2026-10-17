"""
Tests for LegalRetriever ranking, top-k selection, and metadata extraction.
"""

import pytest
from langchain_core.documents import Document
from rag_assistant.embeddings import EmbeddingManager
from rag_assistant.retriever import LegalRetriever, RetrievedPassage
from rag_assistant.vectorstore import FAISSVectorStoreManager


@pytest.fixture(scope="module")
def shared_embeddings():
    return EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu").get_embeddings()


@pytest.fixture
def test_retriever(tmp_path, shared_embeddings):
    manager = FAISSVectorStoreManager(index_dir=tmp_path / "idx", embeddings=shared_embeddings)
    docs = [
        Document(
            page_content="Clause 18 mandates arbitration in Hyderabad under Arbitration and Conciliation Act.",
            metadata={"filename": "bluewave_contract.pdf", "page": 4, "chunk_id": 10, "source": "bluewave_contract.pdf"}
        ),
        Document(
            page_content="Anil Kumar alleged wrongful termination and claimed Rs. 15 lakhs severance pay.",
            metadata={"filename": "anil_kumar_claim.pdf", "page": 2, "chunk_id": 20, "source": "anil_kumar_claim.pdf"}
        ),
        Document(
            page_content="The registered gift settlement deed was executed in 2012 by late K. Venkataiah.",
            metadata={"filename": "lakshmidevi_deed.pdf", "page": 1, "chunk_id": 30, "source": "lakshmidevi_deed.pdf"}
        ),
    ]
    manager.build_index(docs)
    return LegalRetriever(vector_store_manager=manager, k=2, use_reranker=False)


def test_retriever_empty_query(test_retriever):
    passages = test_retriever.retrieve("")
    assert passages == []

    passages_ws = test_retriever.retrieve("   ")
    assert passages_ws == []


def test_retriever_query_matching(test_retriever):
    passages = test_retriever.retrieve("Where is the arbitration seat located?", top_k=1)
    assert len(passages) == 1
    top = passages[0]
    assert top.filename == "bluewave_contract.pdf"
    assert top.page == 4
    assert top.chunk_id == 10
    assert "arbitration" in top.content.lower()
    assert top.score > 0.0

    dict_repr = top.to_dict()
    assert dict_repr["filename"] == "bluewave_contract.pdf"
    assert dict_repr["page"] == 4
