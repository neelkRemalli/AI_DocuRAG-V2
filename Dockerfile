FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Binary wheels are used for the pinned FAISS and PyMuPDF dependencies.
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

RUN mkdir -p /app/data/faiss_index /app/documents/docs-pdf
VOLUME ["/app/data/faiss_index", "/app/documents/docs-pdf"]
EXPOSE 8000
CMD ["uvicorn", "rag_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
