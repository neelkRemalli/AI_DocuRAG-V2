"""Construct the embedding client."""

from langchain_openai import OpenAIEmbeddings


def create_embedding_model(api_key: str, base_url: str, model_name: str) -> OpenAIEmbeddings:
    """Create an OpenAI-compatible embeddings client."""

    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url)
