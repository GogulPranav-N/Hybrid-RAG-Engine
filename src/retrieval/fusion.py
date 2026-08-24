"""Reciprocal Rank Fusion (RRF) to merge dense and sparse ranked lists."""

from __future__ import annotations

import logging
from collections import defaultdict

from src.models import ScoredChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    *ranked_lists: list[ScoredChunk],
    k: int = 60,
) -> list[ScoredChunk]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    For each document d, its fused score is:
        RRF(d) = Σ_m  1 / (k + rank_m(d))

    where m iterates over each search method and rank_m(d) is the
    1-based rank of document d in method m's result list.

    Args:
        *ranked_lists: Variable number of ranked result lists (each pre-sorted
                       by score descending).
        k: Smoothing constant (default 60). Prevents any single top result
           from dominating the fused score.

    Returns:
        Merged list of ScoredChunks sorted by fused RRF score (descending).
        Each chunk's score is replaced with its fused score, and source_method
        is set to "fused".
    """
    # chunk_id → accumulated RRF score
    fused_scores: dict[str, float] = defaultdict(float)

    # chunk_id → best Chunk object (keep the one with most metadata)
    chunk_lookup: dict[str, ScoredChunk] = {}

    for ranked_list in ranked_lists:
        for rank_0based, scored_chunk in enumerate(ranked_list):
            chunk_id = scored_chunk.chunk.id
            rank_1based = rank_0based + 1

            fused_scores[chunk_id] += 1.0 / (k + rank_1based)

            # Keep the version with the most complete metadata
            if chunk_id not in chunk_lookup or (
                scored_chunk.chunk.source and not chunk_lookup[chunk_id].chunk.source
            ):
                chunk_lookup[chunk_id] = scored_chunk

    # Build fused results
    fused_results: list[ScoredChunk] = []
    for chunk_id, fused_score in fused_scores.items():
        original = chunk_lookup[chunk_id]
        fused_results.append(
            ScoredChunk(
                chunk=original.chunk,
                score=fused_score,
                source_method="fused",
            )
        )

    # Sort by fused score descending
    fused_results.sort(key=lambda sc: sc.score, reverse=True)

    logger.info(
        "RRF fusion: merged %d lists → %d unique chunks",
        len(ranked_lists),
        len(fused_results),
    )
    return fused_results
