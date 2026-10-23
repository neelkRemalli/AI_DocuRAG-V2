"""Load PDF pages into LangChain Document objects."""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_documents(documents_dir: Path) -> list[Document]:
    """Load all PDFs in a directory and preserve source/page metadata."""

    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")
    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {documents_dir}")

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        for page in PyMuPDFLoader(str(pdf_path)).load():
            if page.page_content.strip():
                documents.append(page)

    if not documents:
        raise ValueError(f"No extractable text found in PDFs under: {documents_dir}")
    return documents
