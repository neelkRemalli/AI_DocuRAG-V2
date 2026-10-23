"""Construct the chat model and generate answers."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def create_llm(api_key: str, base_url: str, model_name: str) -> ChatOpenAI:
    """Create an OpenAI-compatible chat-model client."""

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def generate_answer(
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    question: str,
    context: str,
) -> str:
    """Format the prompt, invoke the model, and return answer text."""

    if not context.strip():
        return "I don't know."
    messages = prompt.format_messages(question=question, context=context)
    response = llm.invoke(messages)
    return str(response.content)
