"""Tests for chunking while preserving document metadata."""

import pytest
from langchain_core.documents import Document

from rag_app.document_splitter import split_documents


def test_split_documents_preserves_metadata_and_adds_start_index():
    document = Document(
        page_content="First sentence. Second sentence. Third sentence.",
        metadata={"source": "guide.pdf", "page": 3},
    )

    chunks = split_documents([document], chunk_size=20, chunk_overlap=4)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "guide.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_split_documents_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents([Document(page_content="Text")], 10, 10)
