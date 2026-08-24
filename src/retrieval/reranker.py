"""Cross-encoder reranking to refine retrieval results."""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from src.config import settings
from src.models import ScoredChunk

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    scored_chunks: list[ScoredChunk],
    cross_encoder: CrossEncoder,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """
    Rerank candidate chunks using a cross-encoder model.

    The cross-encoder processes (query, chunk_text) pairs jointly through
    a transformer, producing a much more precise relevance score than
    bi-encoder dot-product similarity.

    Args:
        query: The user's question.
        scored_chunks: Candidate chunks from retrieval/fusion (pre-filtered).
        cross_encoder: Loaded CrossEncoder model instance.
        top_k: Number of top results to keep after reranking.

    Returns:
        Top-k chunks sorted by cross-encoder score (descending).
    """
    top_k = top_k or settings.top_k_rerank

    if not scored_chunks:
        return []

    # Build (query, document) pairs for the cross-encoder
    pairs = [(query, sc.chunk.text) for sc in scored_chunks]

    # Score all pairs
    scores = cross_encoder.predict(pairs)

    # Attach cross-encoder scores and re-sort
    reranked: list[ScoredChunk] = []
    for i, sc in enumerate(scored_chunks):
        reranked.append(
            ScoredChunk(
                chunk=sc.chunk,
                score=float(scores[i]),
                source_method="reranked",
            )
        )

    reranked.sort(key=lambda sc: sc.score, reverse=True)

    # Keep only top-k
    result = reranked[:top_k]

    logger.info(
        "Reranking: %d candidates → top %d (best score: %.4f)",
        len(scored_chunks),
        len(result),
        result[0].score if result else 0.0,
    )
    return result
