"""Tests for configuration validation."""

import pytest

from rag_app.config import load_settings


def test_load_settings_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        load_settings()


def test_load_settings_rejects_invalid_overlap(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        load_settings()


def test_load_settings_accepts_valid_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_SIZE", "400")
    monkeypatch.setenv("CHUNK_OVERLAP", "80")
    monkeypatch.setenv("RETRIEVAL_K", "2")
    settings = load_settings()
    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 80
    assert settings.retrieval_k == 2


def test_load_settings_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_SIZE", "large")
    with pytest.raises(ValueError, match="CHUNK_SIZE must be an integer"):
        load_settings()
