#!/usr/bin/env python3
"""CLI script to run document ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import run_indexing
from src.ingestion.loader import load_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument(
        "--data-dir",
        default=str(settings.data_dir),
        help=f"Directory containing documents (default: {settings.data_dir})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Chunk size in tokens (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Chunk overlap in tokens (default: {settings.chunk_overlap})",
    )
    args = parser.parse_args()

    logger.info("Loading documents from: %s", args.data_dir)
    documents = load_directory(args.data_dir)
    if not documents:
        logger.error("No documents found in %s", args.data_dir)
        sys.exit(1)

    logger.info("Chunking %d documents...", len(documents))
    chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)

    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedding_model = SentenceTransformer(settings.embedding_model)

    logger.info("Connecting to Qdrant: %s", settings.qdrant_url)
    qdrant_client = QdrantClient(url=settings.qdrant_url)

    logger.info("Indexing %d chunks...", len(chunks))
    run_indexing(chunks, embedding_model, qdrant_client)

    logger.info("✅ Ingestion complete!")
    logger.info("   Documents: %d", len(documents))
    logger.info("   Chunks:    %d", len(chunks))
    logger.info("   Sources:   %s", [d.source_path for d in documents])

    qdrant_client.close()


if __name__ == "__main__":
    main()
