"""Centralized application configuration and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable settings required by the application."""

    openai_api_key: str
    openai_base_url: str
    embedding_model: str
    chat_model: str
    documents_dir: Path
    vector_store_dir: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int


def _read_integer(name: str, default: str) -> int:
    value = os.getenv(name, default)
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def load_settings() -> Settings:
    """Read environment variables, convert values, and validate invariants."""

    chunk_size = _read_integer("CHUNK_SIZE", "400")
    chunk_overlap = _read_integer("CHUNK_OVERLAP", "80")
    retrieval_k = _read_integer("RETRIEVAL_K", "2")

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
    if retrieval_k <= 0:
        raise ValueError("RETRIEVAL_K must be greater than 0.")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    return Settings(
        openai_api_key=api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        documents_dir=PROJECT_ROOT / "documents" / "docs-pdf",
        vector_store_dir=PROJECT_ROOT / "data" / "faiss_index",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=retrieval_k,
    )
