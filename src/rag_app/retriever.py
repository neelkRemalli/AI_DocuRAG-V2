"""Retrieve relevant document chunks."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def retrieve_documents(vector_store: FAISS, query: str, k: int) -> list[Document]:
    """Return the k most relevant documents for a question."""

    if not query or not query.strip():
        raise ValueError("query must not be empty.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    return vector_store.similarity_search(query, k=k)
