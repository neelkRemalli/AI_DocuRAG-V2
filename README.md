# AI_DocuRAG V2

AI_DocuRAG V2 is a document question-answering application built with LangChain, OpenAI-compatible embedding and chat models, and a local FAISS vector index. It provides both a command-line interface and a FastAPI service over the same retrieval-augmented generation (RAG) workflow.

## Features

- Load and split text from PDF documents.
- Store document embeddings locally in FAISS for reuse between runs.
- Retrieve relevant document passages and generate context-grounded answers.
- Use the same RAG service from the CLI and HTTP API.
- Run offline unit tests with mocked services and deterministic local embeddings.

## Requirements

- Python 3.11 or newer
- An OpenAI-compatible API key and models that support embeddings and chat completions
- Text-based PDF files (OCR is not included)

## Quick start

Create a virtual environment, install the application, and prepare local configuration:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. Add PDF files to `documents/docs-pdf/`.

Run the CLI:

```bash
docurag
```

On the first run, the application reads the PDFs, creates text chunks and embeddings, and saves the index to `data/faiss_index/`. Later runs reuse the saved index.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required | API credential for the configured provider. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API endpoint. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model used to index and search documents. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model used to answer questions. |
| `CHUNK_SIZE` | `400` | Maximum chunk size in characters. |
| `CHUNK_OVERLAP` | `80` | Character overlap between adjacent chunks. Must be less than `CHUNK_SIZE`. |
| `RETRIEVAL_K` | `2` | Number of chunks retrieved per question. |

Copy `.env.example` to `.env` for local development. Never commit `.env` or put real credentials in source control.

## HTTP API

Start the API server:

```bash
uvicorn rag_app.api:app --host 0.0.0.0 --port 8000
```

Check process liveness:

```bash
curl http://localhost:8000/health
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What do the documents say about the project?"}'
```

The ask endpoint accepts a `query` string from 1 to 4,000 characters and returns an `answer` string. If retrieval provides no usable context, the application answers `I don't know.` The health endpoint is a liveness check; it does not test provider connectivity. The API initializes the RAG runtime at startup and needs a valid API key plus a saved index or PDFs to build one.

## Index maintenance

The saved index is reused on startup. If you change the PDF collection or embedding model, remove the generated files in `data/faiss_index/` before restarting to rebuild it. Only load index files you trust: LangChain's FAISS persistence uses pickle for its document store.

## Docker

Build the image:

```bash
docker build -t ai-docurag-v2 .
```

Run it with your local configuration and persistent mounts for PDFs and the FAISS index:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/documents/docs-pdf:/app/documents/docs-pdf:ro" \
  -v "$(pwd)/data/faiss_index:/app/data/faiss_index" \
  ai-docurag-v2
```

The image does not contain local documents or a prebuilt index. Provide PDFs on the first run and retain the index volume between container runs. Keep secrets outside the image.

## Tests

Install development dependencies and run the offline test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The tests cover configuration, context construction, chunking, service orchestration, API behavior, and FAISS persistence. They use test doubles and local deterministic embeddings; they do not require a real API key or make provider requests. CI runs the same suite on pushes and pull requests.
