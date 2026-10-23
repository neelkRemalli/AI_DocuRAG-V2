"""HTTP API exposing the existing RAG service."""

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .runtime import RagRuntime, initialize_rag
from .service import answer_question

logger = logging.getLogger(__name__)
RuntimeInitializer = Callable[[], RagRuntime]


class AskRequest(BaseModel):
    """Validated request body for a question."""

    query: Annotated[str, Field(min_length=1, max_length=4000)]


class AskResponse(BaseModel):
    """Stable JSON response contract."""

    answer: str


def create_app(initializer: RuntimeInitializer = initialize_rag) -> FastAPI:
    """Create an API app, with an injectable initializer for offline tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_runtime = initializer()
        yield

    application = FastAPI(
        title="AI_DocuRAG V2 API",
        description="Question-answering API over configured PDF documents.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        """Return a liveness response; this does not check provider availability."""

        return {"status": "ok"}

    @application.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, http_request: Request) -> AskResponse:
        """Answer a question using the initialized RAG resources."""

        runtime: RagRuntime = http_request.app.state.rag_runtime
        try:
            answer = answer_question(
                query=request.query,
                db=runtime.db,
                llm=runtime.llm,
                prompt=runtime.prompt,
                retrieval_k=runtime.retrieval_k,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            logger.exception("RAG request failed.")
            raise HTTPException(
                status_code=502,
                detail="The question could not be answered right now.",
            ) from error
        return AskResponse(answer=answer)

    return application


app = create_app()
