"""Sparse retrieval via BM25."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """Wrapper around a pickled BM25 index with chunk metadata."""

    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str], chunk_texts: list[str]):
        self.bm25 = bm25
        self.chunk_ids = chunk_ids
        self.chunk_texts = chunk_texts

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BM25Index":
        """Load a previously saved BM25 index from disk."""
        path = Path(path or settings.bm25_index_path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}. Run ingestion first.")

        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            bm25=data["bm25"],
            chunk_ids=data["chunk_ids"],
            chunk_texts=data["chunk_texts"],
        )


def sparse_search(
    query: str,
    bm25_index: BM25Index,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """
    Search the BM25 index with a text query.

    Returns scored chunks sorted by BM25 score (highest first).
    """
    top_k = top_k or settings.top_k_retrieval

    # Tokenize query the same way we tokenized the corpus
    tokenized_query = query.lower().split()

    # Get scores for all documents
    scores = bm25_index.bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    scored_chunks: list[ScoredChunk] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue  # Skip zero-score results

        chunk = Chunk(
            id=bm25_index.chunk_ids[idx],
            text=bm25_index.chunk_texts[idx],
            source="",  # BM25 index doesn't store full metadata
            chunk_index=idx,
        )
        scored_chunks.append(
            ScoredChunk(
                chunk=chunk,
                score=score,
                source_method="sparse",
            )
        )

    logger.info("Sparse search returned %d results for query: %.50s...", len(scored_chunks), query)
    return scored_chunks
