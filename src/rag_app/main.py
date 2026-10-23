"""Command-line entry point for the RAG flow."""

import logging

from .runtime import initialize_rag
from .service import answer_question


def main() -> None:
    """Initialize RAG once, then answer one command-line question."""

    logging.basicConfig(level=logging.INFO)
    runtime = initialize_rag()
    question = input("Ask a question about your documents: ").strip()
    if not question:
        print("Please enter a non-empty question.")
        return
    answer = answer_question(
        question, runtime.db, runtime.llm, runtime.prompt, runtime.retrieval_k
    )
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
