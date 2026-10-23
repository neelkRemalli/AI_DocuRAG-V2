"""Initialize shared resources required by the RAG application."""

import logging
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings, load_settings
from .embeddings import create_embedding_model
from .llm import create_llm
from .prompt import create_rag_prompt
from .vector_store import load_or_create_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagRuntime:
    """Resources shared by CLI/API requests."""

    db: FAISS
    llm: ChatOpenAI
    prompt: ChatPromptTemplate
    retrieval_k: int


def initialize_rag(settings: Settings | None = None) -> RagRuntime:
    """Validate settings and initialize the shared RAG resources."""

    settings = settings or load_settings()
    logger.info("Initializing embedding client and vector store.")
    embeddings: OpenAIEmbeddings = create_embedding_model(
        settings.openai_api_key, settings.openai_base_url, settings.embedding_model
    )
    db = load_or_create_vector_store(
        settings.documents_dir,
        settings.vector_store_dir,
        embeddings,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    llm = create_llm(settings.openai_api_key, settings.openai_base_url, settings.chat_model)
    return RagRuntime(db=db, llm=llm, prompt=create_rag_prompt(), retrieval_k=settings.retrieval_k)
