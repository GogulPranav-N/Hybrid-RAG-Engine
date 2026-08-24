"""LLM generation via Groq API with sync and streaming support."""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from groq import AsyncGroq, Groq

from src.config import settings
from src.models import ScoredChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful technical assistant. Answer the user's question using ONLY \
the provided context chunks below. If the context does not contain enough \
information to answer the question, say so clearly — do not make up information.

For each claim in your answer, cite the source using [Source: <filename>] format.

Keep your answers concise, accurate, and well-structured with markdown formatting.
"""


def _build_context_block(chunks: list[ScoredChunk]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    parts: list[str] = []
    for i, sc in enumerate(chunks, 1):
        source = sc.chunk.source or "unknown"
        # Extract just the filename from the path
        filename = source.rsplit("/", 1)[-1] if "/" in source else source
        parts.append(
            f"--- Chunk {i} [Source: {filename}] ---\n{sc.chunk.text}\n"
        )
    return "\n".join(parts)


def _build_messages(question: str, chunks: list[ScoredChunk]) -> list[dict]:
    """Build the chat messages array for the Groq API."""
    context = _build_context_block(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer based only on the context above, with citations."
            ),
        },
    ]


def generate(
    question: str,
    chunks: list[ScoredChunk],
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """
    Generate a complete answer using Groq (synchronous).

    Args:
        question: The user's question.
        chunks: Retrieved context chunks.
        api_key: Groq API key (defaults to settings).
        model: LLM model name (defaults to settings).

    Returns:
        The complete generated answer string.
    """
    client = Groq(api_key=api_key or settings.groq_api_key)
    messages = _build_messages(question, chunks)

    response = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=0.1,  # Low temperature for factual RAG answers
        max_tokens=1024,
    )

    answer = response.choices[0].message.content or ""
    logger.info(
        "Generated answer: %d tokens, model=%s",
        response.usage.completion_tokens if response.usage else 0,
        model or settings.llm_model,
    )
    return answer


async def generate_async(
    question: str,
    chunks: list[ScoredChunk],
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """
    Generate a complete answer using Groq asynchronously.
    """
    client = AsyncGroq(api_key=api_key or settings.groq_api_key)
    messages = _build_messages(question, chunks)

    response = await client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content or ""
    return answer


async def generate_stream(
    question: str,
    chunks: list[ScoredChunk],
    api_key: str | None = None,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream answer tokens from Groq (async generator for SSE).

    Yields individual text chunks as they arrive from the API.
    """
    client = AsyncGroq(api_key=api_key or settings.groq_api_key)
    messages = _build_messages(question, chunks)

    stream = await client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content is not None:
            yield delta.content
