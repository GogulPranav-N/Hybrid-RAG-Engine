"""Shared singleton dependencies loaded once at application startup."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import settings
from src.retrieval.sparse import BM25Index

logger = logging.getLogger(__name__)


class AppState:
    """
    Holds all heavyweight singleton objects.

    Initialized during FastAPI lifespan startup, torn down on shutdown.
    """

    def __init__(self):
        self.embedding_model: SentenceTransformer | None = None
        self.cross_encoder: CrossEncoder | None = None
        self.qdrant_client: QdrantClient | None = None
        self.bm25_index: BM25Index | None = None

    def startup(self) -> None:
        """Load all models and connect to services."""
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self.embedding_model = SentenceTransformer(settings.embedding_model)

        logger.info("Loading cross-encoder: %s", settings.reranker_model)
        self.cross_encoder = CrossEncoder(settings.reranker_model)

        logger.info("Connecting to Qdrant: %s", settings.qdrant_url)
        try:
            client = QdrantClient(url=settings.qdrant_url, timeout=3.0)
            client.get_collections()
            self.qdrant_client = client
            logger.info("Connected to remote Qdrant service at %s", settings.qdrant_url)
        except Exception as e:
            logger.warning("Could not connect to Qdrant service at %s (%s). Falling back to local embedded storage.", settings.qdrant_url, e)
            self.qdrant_client = QdrantClient(path="./qdrant_storage")


        # Try to load BM25 index (may not exist yet before first ingestion)
        try:
            self.bm25_index = BM25Index.load()
            logger.info("BM25 index loaded from %s", settings.bm25_index_path)
        except FileNotFoundError:
            logger.warning("BM25 index not found — run ingestion first")
            self.bm25_index = None

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.qdrant_client:
            self.qdrant_client.close()
            logger.info("Qdrant client closed")


# Global singleton — set during lifespan
app_state = AppState()
