"""Orchestrate one RAG question-answering operation."""

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .context import build_context
from .llm import generate_answer
from .retriever import retrieve_documents


def answer_question(
    query: str,
    db: FAISS,
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    retrieval_k: int,
) -> str:
    """Retrieve evidence and generate a context-grounded answer."""

    if not query or not query.strip():
        raise ValueError("query must not be empty.")
    retrieved_documents = retrieve_documents(db, query, retrieval_k)
    context = build_context(retrieved_documents)
    return generate_answer(llm, prompt, query, context)
