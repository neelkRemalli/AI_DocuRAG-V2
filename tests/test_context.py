"""Tests for retrieved-context construction."""

from langchain_core.documents import Document

from rag_app.context import build_context


def test_build_context_joins_document_text():
    documents = [Document(page_content=" First chunk. "), Document(page_content="Second chunk.")]
    assert build_context(documents) == "First chunk.\n\nSecond chunk."


def test_build_context_returns_empty_string_for_no_documents():
    assert build_context([]) == ""


def test_build_context_ignores_blank_documents():
    assert build_context([Document(page_content=" ")]) == ""
