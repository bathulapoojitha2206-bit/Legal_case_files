"""
Text splitting and chunking module for legal case documents.
Uses RecursiveCharacterTextSplitter with legal-aware separators and metadata preservation.
"""

from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:
    """Splits documents into overlapping chunks with rich metadata."""

    # Separators prioritized for legal case text (sections, clauses, sentences)
    LEGAL_SEPARATORS = [
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        "; ",
        ": ",
        ", ",
        " ",
        ""
    ]

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        if chunk_size <= 50:
            raise ValueError(f"chunk_size must be > 50, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.LEGAL_SEPARATORS,
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of LangChain documents into chunks, retaining and
        enriching metadata with unique chunk IDs and character counts.

        Args:
            documents: List of parsed LangChain Document objects.

        Returns:
            List of chunked Document objects.
        """
        if not documents:
            return []

        chunks: List[Document] = []
        global_chunk_id = 0

        for doc_idx, doc in enumerate(documents):
            doc_chunks = self.splitter.split_documents([doc])
            for local_idx, chunk in enumerate(doc_chunks):
                # Enrich metadata
                meta = dict(chunk.metadata)
                meta["chunk_id"] = global_chunk_id
                meta["chunk_index_in_doc"] = local_idx
                meta["chunk_char_count"] = len(chunk.page_content)
                meta["chunk_preview"] = (
                    chunk.page_content[:60].replace("\n", " ").strip() + "..."
                    if len(chunk.page_content) > 60
                    else chunk.page_content.replace("\n", " ").strip()
                )
                chunk.metadata = meta
                chunks.append(chunk)
                global_chunk_id += 1

        return chunks

    def get_chunking_stats(self, chunks: List[Document]) -> Dict[str, float]:
        """Compute statistical breakdown of generated chunks."""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_char_length": 0.0,
                "min_char_length": 0,
                "max_char_length": 0,
            }

        lengths = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_char_length": round(sum(lengths) / len(lengths), 2),
            "min_char_length": min(lengths),
            "max_char_length": max(lengths),
        }
