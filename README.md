# AI_DocuRAG_V2

### A modular retrieval-augmented generation system for document question answering

AI_DocuRAG_V2 is a Python application that answers questions using information retrieved from PDF documents. It combines document loading and chunking, semantic vector search, context construction, and a language model in a modular pipeline that can be used from a command-line interface or an HTTP API.

## Documentation

The implementation and its responsibilities are documented in the source modules under [`src/rag_app/`](src/rag_app/). Dependency and test configuration is in [`pyproject.toml`](pyproject.toml), and CI is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

The project specification is kept as a local file and intentionally excluded from version control. The checked-in source and configuration files describe the implementation available in this repository.

## Project goals

- Build a structured PDF question-answering pipeline.
- Retrieve relevant document content with semantic vector search.
- Generate context-grounded answers with a large language model (LLM).
- Separate responsibilities into small, maintainable modules.
- Expose the RAG workflow through a CLI and HTTP API.
- Support automated tests and consistent development practices.
- Prepare the application for containerized deployment.

## Architecture

```text
PDF documents
      |
      v
Document loader -> Text splitter -> Embedding model -> FAISS vector store
                                                     |
Question -> Retriever -> Context builder -> Prompt + LLM -> Answer
```

The CLI and HTTP API both call the same RAG service. Shared models and the vector store are initialized once when each application process starts.

## Technology stack

| Component | Technology |
| --- | --- |
| Language | Python 3.11+ |
| LLM integration | LangChain and an OpenAI-compatible API |
| Embeddings | OpenAI-compatible embedding model |
| Vector store | FAISS CPU |
| PDF processing | PyMuPDF through LangChain Community loaders |
| HTTP API | FastAPI and Uvicorn |
| Validation | Pydantic |
| Tests | pytest |
| Containerization | Docker |
| CI | GitHub Actions |

Pinned RAG dependency versions and compatible web framework ranges are declared in `pyproject.toml`.

## Project structure

```text
AI_DocuRAG_V2/
├── .dockerignore
├── .github/workflows/ci.yml
├── .gitignore
├── documents/docs-pdf/       # Add local PDF files here; PDFs are gitignored
├── data/faiss_index/         # Generated persistent FAISS index; gitignored
├── src/rag_app/
│   ├── api.py                # FastAPI endpoints
│   ├── config.py             # Environment settings and validation
│   ├── document_loader.py    # PDF loading
│   ├── document_splitter.py  # Text chunking
│   ├── embeddings.py         # Embedding client
│   ├── vector_store.py       # FAISS build, save, and load
│   ├── retriever.py          # Similarity search
│   ├── context.py            # Retrieved context assembly
│   ├── prompt.py              # Grounded-answer prompt
│   ├── llm.py                 # Chat model and answer generation
│   ├── service.py             # RAG request orchestration
│   ├── runtime.py             # Shared resource initialization
│   └── main.py                # CLI entry point
├── tests/
├── .env.example
├── Dockerfile
└── pyproject.toml
```

## Getting started

### 1. Create an environment and install dependencies

```bash
git clone https://github.com/neelkRemalli/AI_DocuRAG-V2.git
cd AI_DocuRAG-V2
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the application and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

### 2. Configure the application

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and add PDF files to `documents/docs-pdf/`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required | Credential for the configured API provider. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API endpoint. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model used to embed document chunks and queries. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Model used to generate answers. |
| `CHUNK_SIZE` | `400` | Maximum chunk size in characters. |
| `CHUNK_OVERLAP` | `80` | Character overlap between adjacent chunks. |
| `RETRIEVAL_K` | `2` | Number of chunks retrieved for a question. |

Never commit `.env`, API keys, or other credentials. The example file intentionally contains no secret.

### 3. Run the CLI

```bash
docurag
```

The first run extracts PDF text, creates chunks and embeddings, and saves a FAISS index under `data/faiss_index/`. Later runs reuse the saved index. If the PDFs or embedding model change, remove the generated index before restarting to rebuild it. The PDF loader extracts text; OCR for scanned documents is not included.

## HTTP API

Start the API server:

```bash
uvicorn rag_app.api:app --host 0.0.0.0 --port 8000
```

Liveness check:

```bash
curl http://localhost:8000/health
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What do the documents say about the project?"}'
```

`POST /ask` accepts a `query` string between 1 and 4,000 characters and returns an `answer`. If retrieved context is empty, the application responds with `I don't know.` The health endpoint reports process liveness and does not check whether an external provider is reachable. API startup requires a valid configuration and either an existing index or PDFs from which to build one.

## Docker

Build the image:

```bash
docker build -t ai-docurag-v2 .
```

Run it with environment configuration and persistent document and index mounts:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/documents/docs-pdf:/app/documents/docs-pdf:ro" \
  -v "$(pwd)/data/faiss_index:/app/data/faiss_index" \
  ai-docurag-v2
```

The image does not include local PDFs or an index. Supply PDFs on the first run and retain the index volume between container runs. Do not copy `.env` or credentials into the image.

## Tests and verification

Run the offline test suite:

```bash
python -m pytest
```

The tests cover configuration, context creation, splitting, service orchestration, API behavior, and FAISS persistence. They use mocks and deterministic local embeddings, so they do not need a real API key or make paid provider calls. CI installs `.[dev]` and runs the same test suite.

The implementation has passed 18 automated tests, package consistency checks, import checks, and a Docker image build. Live PDF indexing and question answering against an external provider require local documents and credentials and were not part of the automated verification.

## Security notes

- Keep API credentials in local environment configuration, never in source control.
- FAISS persistence uses pickle for the document store. Only load index files from a trusted source.
- Prompt instructions help keep answers grounded but do not eliminate hallucinations or prompt-injection risk.

## License

No license has been specified. Add a `LICENSE` file before distributing the project under an open-source license.
