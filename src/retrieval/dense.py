"""Dense retrieval via Qdrant vector search."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


def dense_search(
    query: str,
    embedding_model: SentenceTransformer,
    qdrant_client: QdrantClient,
    top_k: int | None = None,
    collection_name: str | None = None,
) -> list[ScoredChunk]:
    """
    Embed the query and perform cosine similarity search in Qdrant.

    Returns scored chunks sorted by relevance (highest first).
    """
    top_k = top_k or settings.top_k_retrieval
    collection = collection_name or settings.collection_name

    # Embed query — BGE models recommend normalization
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    # Search Qdrant
    results = qdrant_client.query_points(
        collection_name=collection,
        query=query_embedding,
        limit=top_k,
    ).points

    scored_chunks: list[ScoredChunk] = []
    for hit in results:
        payload = hit.payload or {}
        chunk = Chunk(
            id=payload.get("chunk_id", str(hit.id)),
            text=payload.get("text", ""),
            source=payload.get("source", ""),
            chunk_index=payload.get("chunk_index", 0),
            token_count=payload.get("token_count", 0),
            metadata={
                k: v
                for k, v in payload.items()
                if k not in ("chunk_id", "text", "source", "chunk_index", "token_count")
            },
        )
        scored_chunks.append(
            ScoredChunk(
                chunk=chunk,
                score=hit.score,
                source_method="dense",
            )
        )

    logger.info("Dense search returned %d results for query: %.50s...", len(scored_chunks), query)
    return scored_chunks
