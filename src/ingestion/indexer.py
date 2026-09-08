"""Index chunks into Qdrant (dense) and build a BM25 index (sparse)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models import Chunk

logger = logging.getLogger(__name__)


def _embed_chunks(
    chunks: Sequence[Chunk],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed chunk texts in batches."""
    texts = [chunk.text for chunk in chunks]
    logger.info("Embedding %d chunks in batches of %d...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # BGE models recommend normalization
    )
    return embeddings.tolist()


def index_to_qdrant(
    chunks: Sequence[Chunk],
    embeddings: list[list[float]],
    client: QdrantClient,
    collection_name: str | None = None,
) -> None:
    """Create/recreate a Qdrant collection and upsert all chunk embeddings."""
    collection = collection_name or settings.collection_name
    vector_size = len(embeddings[0])

    # Recreate collection for clean state
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    logger.info(
        "Created Qdrant collection '%s' (dim=%d, cosine)",
        collection,
        vector_size,
    )

    # Upsert in batches
    batch_size = 100
    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "chunk_id": chunks[i].id,
                "text": chunks[i].text,
                "source": chunks[i].source,
                "chunk_index": chunks[i].chunk_index,
                "token_count": chunks[i].token_count,
                **chunks[i].metadata,
            },
        )
        for i in range(len(chunks))
    ]

    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection, points=batch)

    logger.info("Upserted %d points to Qdrant collection '%s'", len(points), collection)


def build_bm25_index(
    chunks: Sequence[Chunk],
    save_path: str | Path | None = None,
) -> BM25Okapi:
    """
    Build a BM25 index from chunk texts and optionally save to disk.

    Also saves the chunk-ID-to-index mapping alongside the BM25 index.
    """
    # Tokenize: lowercase + split on whitespace (simple but effective)
    tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save BM25 index + chunk mapping together
        data = {
            "bm25": bm25,
            "chunk_ids": [chunk.id for chunk in chunks],
            "chunk_texts": [chunk.text for chunk in chunks],
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved BM25 index to %s", save_path)

    return bm25


def run_indexing(
    chunks: Sequence[Chunk],
    embedding_model: SentenceTransformer,
    qdrant_client: QdrantClient,
) -> None:
    """Full indexing pipeline: embed → Qdrant + BM25."""
    # 1. Embed
    embeddings = _embed_chunks(chunks, embedding_model)

    # 2. Index to Qdrant (dense)
    index_to_qdrant(chunks, embeddings, qdrant_client)

    # 3. Build BM25 index (sparse)
    build_bm25_index(chunks, save_path=settings.bm25_index_path)

    logger.info("Indexing complete: %d chunks indexed", len(chunks))
