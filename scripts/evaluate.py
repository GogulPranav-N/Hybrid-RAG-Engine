#!/usr/bin/env python3
"""CLI script to run evaluation across all RAG modes."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import settings
from src.evaluation.compare import run_comparison
from src.generation.llm import generate
from src.models import QueryRequest, SearchMode, ScoredChunk
from src.retrieval.dense import dense_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.retrieval.sparse import BM25Index, sparse_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline configurations")
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output file for results (default: evaluation_results.json)",
    )
    args = parser.parse_args()

    # Load models
    logger.info("Loading models...")
    embedding_model = SentenceTransformer(settings.embedding_model)
    cross_encoder = CrossEncoder(settings.reranker_model)
    qdrant_client = QdrantClient(url=settings.qdrant_url)
    bm25_index = BM25Index.load()

    async def query_fn(request: QueryRequest) -> tuple[str, list[ScoredChunk]]:
        """Pipeline function for evaluation."""
        # Dense search
        dense_results = dense_search(
            query=request.question,
            embedding_model=embedding_model,
            qdrant_client=qdrant_client,
            top_k=20,
        )

        if request.mode == SearchMode.NAIVE:
            chunks = dense_results[: request.top_k]
        elif request.mode == SearchMode.HYBRID:
            sparse_results = sparse_search(
                query=request.question,
                bm25_index=bm25_index,
                top_k=20,
            )
            fused = reciprocal_rank_fusion(dense_results, sparse_results)
            chunks = fused[: request.top_k]
        else:  # HYBRID_RERANKED
            sparse_results = sparse_search(
                query=request.question,
                bm25_index=bm25_index,
                top_k=20,
            )
            fused = reciprocal_rank_fusion(dense_results, sparse_results)
            chunks = rerank(
                query=request.question,
                scored_chunks=fused[:25],
                cross_encoder=cross_encoder,
                top_k=request.top_k,
            )

        answer = generate(request.question, chunks)
        return answer, chunks

    # Run comparison
    logger.info("Starting evaluation...")
    comparison = asyncio.run(run_comparison(query_fn, output_path=args.output))

    logger.info("✅ Evaluation complete! Results saved to %s", args.output)

    qdrant_client.close()


if __name__ == "__main__":
    main()
