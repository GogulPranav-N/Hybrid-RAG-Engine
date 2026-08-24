"""Token-aware document chunking with configurable size and overlap."""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import tiktoken

from src.models import Chunk, Document

logger = logging.getLogger(__name__)

# Use the same tokenizer that OpenAI / most modern models use
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def _generate_chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic chunk ID from source path + index."""
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_text_by_tokens(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split text into chunks of approximately `chunk_size` tokens,
    with `chunk_overlap` tokens of overlap between consecutive chunks.

    Splits on paragraph boundaries first, then sentence boundaries,
    falling back to token-level splitting.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_tokens: list[int] = []

    for paragraph in paragraphs:
        paragraph_tokens = _ENCODING.encode(paragraph, disallowed_special=())

        # If a single paragraph exceeds chunk_size, split it further
        if len(paragraph_tokens) > chunk_size:
            # Split long paragraphs by sentences
            sentences = paragraph.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sent_tokens = _ENCODING.encode(sentence, disallowed_special=())

                if len(current_tokens) + len(sent_tokens) > chunk_size and current_tokens:
                    # Flush current chunk
                    chunks.append(_ENCODING.decode(current_tokens))
                    # Keep overlap
                    current_tokens = current_tokens[-chunk_overlap:] if chunk_overlap > 0 else []

                current_tokens.extend(sent_tokens)
        else:
            if len(current_tokens) + len(paragraph_tokens) > chunk_size and current_tokens:
                chunks.append(_ENCODING.decode(current_tokens))
                current_tokens = current_tokens[-chunk_overlap:] if chunk_overlap > 0 else []

            current_tokens.extend(paragraph_tokens)

    # Don't forget the last chunk
    if current_tokens:
        chunks.append(_ENCODING.decode(current_tokens))

    return chunks


def chunk_document(
    doc: Document,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a single Document into token-bounded Chunks."""
    text_chunks = _split_text_by_tokens(doc.text, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for i, text in enumerate(text_chunks):
        token_count = _token_count(text)
        chunk_id = _generate_chunk_id(doc.source_path, i)

        chunks.append(
            Chunk(
                id=chunk_id,
                text=text,
                source=doc.source_path,
                chunk_index=i,
                token_count=token_count,
                metadata={
                    **doc.metadata,
                    "file_type": doc.file_type.value,
                },
            )
        )

    return chunks


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split multiple Documents into Chunks."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
