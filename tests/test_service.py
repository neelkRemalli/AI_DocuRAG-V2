"""Unit tests for the RAG service orchestration."""

from unittest.mock import Mock

import pytest

from rag_app.service import answer_question


def test_answer_question_rejects_blank_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        answer_question("   ", Mock(), Mock(), Mock(), 2)


def test_answer_question_connects_retrieval_context_and_generation(monkeypatch):
    import rag_app.service as service

    retrieved = [Mock(page_content="Evidence from the document.")]
    monkeypatch.setattr(service, "retrieve_documents", lambda vector_store, query, k: retrieved)
    monkeypatch.setattr(service, "build_context", lambda documents: "Evidence from the document.")
    monkeypatch.setattr(service, "generate_answer", lambda llm, prompt, question, context: "Generated answer.")

    result = answer_question("What is in the document?", Mock(), Mock(), Mock(), 2)
    assert result == "Generated answer."
