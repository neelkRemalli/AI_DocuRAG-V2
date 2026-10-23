"""Offline tests for building and reloading a persisted FAISS index."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_app.vector_store import load_or_create_vector_store


class DeterministicEmbeddings(Embeddings):
    """Small local embedding stub that never calls an external provider."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(text.encode("utf-8")) % 101), 1.0]


def test_vector_store_builds_then_reloads_without_reindexing(tmp_path, monkeypatch):
    from rag_app import document_loader

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    index_dir = tmp_path / "index"
    embeddings = DeterministicEmbeddings()
    source_documents = [
        Document(page_content="First searchable passage.", metadata={"source": "a.pdf"}),
        Document(page_content="Second searchable passage.", metadata={"source": "b.pdf"}),
    ]
    monkeypatch.setattr(document_loader, "load_documents", lambda _: source_documents)

    first_store = load_or_create_vector_store(
        documents_dir, index_dir, embeddings, chunk_size=100, chunk_overlap=10
    )
    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "index.pkl").is_file()

    def unexpected_reload(_):
        raise AssertionError("Persisted index should load without reading source PDFs")

    monkeypatch.setattr(document_loader, "load_documents", unexpected_reload)
    second_store = load_or_create_vector_store(
        documents_dir, index_dir, embeddings, chunk_size=100, chunk_overlap=10
    )

    assert first_store.index.ntotal == second_store.index.ntotal == 2
    assert second_store.similarity_search("searchable passage", k=1)[0].metadata["source"]


def test_incomplete_vector_store_rebuilds(tmp_path, monkeypatch):
    from rag_app import document_loader

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.faiss").touch()
    monkeypatch.setattr(
        document_loader,
        "load_documents",
        lambda _: [Document(page_content="Rebuilt content.")],
    )

    store = load_or_create_vector_store(
        documents_dir,
        index_dir,
        DeterministicEmbeddings(),
        chunk_size=100,
        chunk_overlap=10,
    )
    assert store.index.ntotal == 1
    assert (index_dir / "index.pkl").is_file()
