"""Tests for API validation and response behavior without external calls."""

from fastapi.testclient import TestClient

from rag_app.api import create_app
from rag_app.runtime import RagRuntime


def fake_runtime():
    return RagRuntime(db=object(), llm=object(), prompt=object(), retrieval_k=2)


def test_health_endpoint():
    with TestClient(create_app(initializer=fake_runtime)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_uses_initialized_runtime(monkeypatch):
    import rag_app.api as api

    monkeypatch.setattr(api, "answer_question", lambda **kwargs: "Test answer.")
    with TestClient(create_app(initializer=fake_runtime)) as client:
        response = client.post("/ask", json={"query": "What is Python?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Test answer."}


def test_ask_endpoint_rejects_missing_query():
    with TestClient(create_app(initializer=fake_runtime)) as client:
        response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_endpoint_hides_internal_errors(monkeypatch):
    import rag_app.api as api

    def fail(**kwargs):
        raise RuntimeError("private provider detail")

    monkeypatch.setattr(api, "answer_question", fail)
    with TestClient(create_app(initializer=fake_runtime)) as client:
        response = client.post("/ask", json={"query": "question"})
    assert response.status_code == 502
    assert response.json() == {"detail": "The question could not be answered right now."}


def test_app_initializes_runtime_once_per_lifespan():
    calls = []

    def initializer():
        calls.append(True)
        return fake_runtime()

    with TestClient(create_app(initializer=initializer)) as client:
        client.get("/health")
        client.get("/health")
    assert calls == [True]
