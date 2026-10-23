"""Convert retrieved documents into prompt context."""

from langchain_core.documents import Document


def build_context(documents: list[Document]) -> str:
    """Join retrieved document content into one context string."""

    return "\n\n".join(
        document.page_content.strip()
        for document in documents
        if document.page_content.strip()
    )
