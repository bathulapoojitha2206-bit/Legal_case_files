"""
Tests for document loader: PDF, TXT, DOCX, error handling, and unsupported formats.
"""

import tempfile
from pathlib import Path
import pytest
from rag_assistant.loader import DocumentLoader, LoadedDocument, DocumentLoaderError


@pytest.fixture
def loader():
    return DocumentLoader()


def test_load_valid_txt(loader, tmp_path):
    txt_file = tmp_path / "case_brief.txt"
    txt_file.write_text("State of Telangana vs. Vikram Chowdary. Criminal Case No. CC/482/2024.", encoding="utf-8")

    result = loader.load_file(txt_file)
    assert result.success is True
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert "Vikram Chowdary" in doc.page_content
    assert doc.metadata["filename"] == "case_brief.txt"
    assert doc.metadata["file_type"] == "txt"
    assert doc.metadata["page"] == 1


def test_load_nonexistent_file(loader, tmp_path):
    missing_file = tmp_path / "does_not_exist.pdf"
    result = loader.load_file(missing_file)
    assert result.success is False
    assert "File not found" in (result.error_message or "")


def test_load_empty_file(loader, tmp_path):
    empty_file = tmp_path / "empty_case.txt"
    empty_file.write_text("", encoding="utf-8")

    result = loader.load_file(empty_file)
    assert result.success is False
    assert "empty" in (result.warning_message or "").lower()


def test_load_unsupported_format(loader, tmp_path):
    bad_file = tmp_path / "data_records.csv"
    bad_file.write_text("id,case_name,status\n1,Chowdary,active", encoding="utf-8")

    result = loader.load_file(bad_file)
    assert result.success is False
    assert "Unsupported document format" in (result.error_message or "")


def test_load_corrupted_pdf(loader, tmp_path):
    corrupt_pdf = tmp_path / "corrupted_legal_document.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 this is completely corrupted binary garbage not valid pdf syntax")

    result = loader.load_file(corrupt_pdf)
    # The loader must not crash the entire process; it should return failure with message
    assert result.success is False
    assert result.error_message is not None or result.warning_message is not None


def test_load_directory(loader, tmp_path):
    # Create two valid files and one unsupported file
    (tmp_path / "case1.txt").write_text("Case 1 details regarding property.", encoding="utf-8")
    (tmp_path / "case2.txt").write_text("Case 2 details regarding employment dispute.", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("not,loaded", encoding="utf-8")

    docs, results = loader.load_directory(tmp_path)
    assert len(docs) == 2
    assert len(results) == 2  # Only supported files are processed
    assert all(r.success for r in results)


def test_load_bytes(loader):
    data = b"Arbitration agreement under Clause 18 between parties."
    result = loader.load_bytes(data, "arbitration_clause.txt")
    assert result.success is True
    assert len(result.documents) == 1
    assert "Clause 18" in result.documents[0].page_content
