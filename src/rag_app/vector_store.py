"""Build, persist, and load the FAISS vector store."""

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def load_or_create_vector_store(
    documents_dir: Path,
    vector_store_dir: Path,
    embeddings: Embeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> FAISS:
    """Load a trusted saved index or build and persist a new index."""

    from .document_loader import load_documents
    from .document_splitter import split_documents

    index_file = vector_store_dir / "index.faiss"
    docstore_file = vector_store_dir / "index.pkl"
    if index_file.is_file() and docstore_file.is_file():
        logger.info("Loading persisted FAISS index from %s", vector_store_dir)
        # FAISS.load_local unpickles index.pkl; only use files from trusted storage.
        return FAISS.load_local(
            str(vector_store_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if index_file.exists() != docstore_file.exists():
        logger.warning("Incomplete FAISS index found; rebuilding it.")

    logger.info("Building FAISS index from PDFs under %s", documents_dir)
    documents: list[Document] = load_documents(documents_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(vector_store_dir))
    logger.info("Built and saved FAISS index containing %s chunks.", len(chunks))
    return vector_store
