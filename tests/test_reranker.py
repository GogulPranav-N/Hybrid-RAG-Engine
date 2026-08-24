"""Tests for the cross-encoder reranker."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.models import Chunk, ScoredChunk


def _make_scored_chunk(chunk_id: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=chunk_id, text=text, source="test.md", chunk_index=0),
        score=score,
        source_method="fused",
    )


class TestReranker:
    """Integration tests for the cross-encoder reranker."""

    @pytest.fixture(scope="class")
    def cross_encoder(self):
        """Load cross-encoder once for all tests in this class."""
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            pytest.skip("Cross-encoder model not available")

    def test_reranking_changes_order(self, cross_encoder):
        """Reranking should potentially reorder results based on relevance."""
        from src.retrieval.reranker import rerank

        chunks = [
            _make_scored_chunk("a", "Docker is a containerization platform.", 0.9),
            _make_scored_chunk("b", "FastAPI is a Python web framework for building APIs.", 0.8),
            _make_scored_chunk("c", "The weather is nice today.", 0.7),
        ]

        reranked = rerank(
            query="How do I build a REST API with Python?",
            scored_chunks=chunks,
            cross_encoder=cross_encoder,
            top_k=3,
        )

        # The FastAPI chunk should rank higher than the weather chunk
        ids = [sc.chunk.id for sc in reranked]
        assert ids.index("b") < ids.index("c"), (
            "FastAPI chunk should rank higher than irrelevant weather chunk"
        )

    def test_reranking_top_k(self, cross_encoder):
        """Reranking should return only top_k results."""
        from src.retrieval.reranker import rerank

        chunks = [
            _make_scored_chunk("a", "Text A", 0.9),
            _make_scored_chunk("b", "Text B", 0.8),
            _make_scored_chunk("c", "Text C", 0.7),
            _make_scored_chunk("d", "Text D", 0.6),
        ]

        reranked = rerank("query", chunks, cross_encoder, top_k=2)
        assert len(reranked) == 2

    def test_reranking_method_label(self, cross_encoder):
        """All reranked results should have source_method='reranked'."""
        from src.retrieval.reranker import rerank

        chunks = [_make_scored_chunk("a", "Some text", 0.9)]
        reranked = rerank("query", chunks, cross_encoder, top_k=1)
        assert all(sc.source_method == "reranked" for sc in reranked)

    def test_empty_input(self, cross_encoder):
        """Empty input should return empty output."""
        from src.retrieval.reranker import rerank

        reranked = rerank("query", [], cross_encoder, top_k=5)
        assert reranked == []
