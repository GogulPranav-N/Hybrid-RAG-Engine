"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Chunk, ScoredChunk
from src.retrieval.fusion import reciprocal_rank_fusion


def _make_scored_chunk(chunk_id: str, score: float, method: str = "dense") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=chunk_id, text=f"Text for {chunk_id}", source="test.md", chunk_index=0),
        score=score,
        source_method=method,
    )


class TestReciprocalRankFusion:
    """Tests for the RRF fusion function."""

    def test_single_list_ordering(self):
        """Single list should preserve ordering."""
        ranked = [
            _make_scored_chunk("a", 0.9, "dense"),
            _make_scored_chunk("b", 0.7, "dense"),
            _make_scored_chunk("c", 0.5, "dense"),
        ]
        fused = reciprocal_rank_fusion(ranked)
        ids = [sc.chunk.id for sc in fused]
        assert ids == ["a", "b", "c"]

    def test_two_lists_merge(self):
        """Two lists should merge correctly, boosting items that appear in both."""
        dense = [
            _make_scored_chunk("a", 0.9, "dense"),
            _make_scored_chunk("b", 0.7, "dense"),
            _make_scored_chunk("c", 0.5, "dense"),
        ]
        sparse = [
            _make_scored_chunk("b", 5.0, "sparse"),
            _make_scored_chunk("d", 3.0, "sparse"),
            _make_scored_chunk("a", 1.0, "sparse"),
        ]
        fused = reciprocal_rank_fusion(dense, sparse)

        # 'b' appears as rank 2 in dense (1/(60+2)) and rank 1 in sparse (1/(60+1))
        # 'a' appears as rank 1 in dense (1/(60+1)) and rank 3 in sparse (1/(60+3))
        # Both should have higher fused scores than items in only one list
        ids = [sc.chunk.id for sc in fused]
        assert "a" in ids
        assert "b" in ids
        assert "d" in ids

        # Items in both lists should score higher
        fused_dict = {sc.chunk.id: sc.score for sc in fused}
        assert fused_dict["b"] > fused_dict["d"]  # b in both, d in only sparse

    def test_rrf_score_formula(self):
        """Verify the actual RRF scores are correct."""
        list1 = [_make_scored_chunk("x", 1.0)]
        fused = reciprocal_rank_fusion(list1, k=60)
        # Score should be 1/(60+1) = 1/61
        assert abs(fused[0].score - 1 / 61) < 1e-10

    def test_empty_lists(self):
        """Empty lists should return empty results."""
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_all_fused_method(self):
        """All fused results should have source_method='fused'."""
        ranked = [_make_scored_chunk("a", 0.9)]
        fused = reciprocal_rank_fusion(ranked)
        assert all(sc.source_method == "fused" for sc in fused)

    def test_k_parameter(self):
        """Different k values should produce different scores."""
        ranked = [_make_scored_chunk("a", 0.9)]
        fused_k10 = reciprocal_rank_fusion(ranked, k=10)
        fused_k60 = reciprocal_rank_fusion(ranked, k=60)

        # 1/(10+1) > 1/(60+1), so k=10 gives higher score
        assert fused_k10[0].score > fused_k60[0].score

    def test_deduplication(self):
        """Same chunk ID across lists should be deduplicated."""
        list1 = [_make_scored_chunk("a", 0.9, "dense")]
        list2 = [_make_scored_chunk("a", 5.0, "sparse")]
        fused = reciprocal_rank_fusion(list1, list2)
        assert len(fused) == 1
        # Score should be 1/(60+1) + 1/(60+1) = 2/61
        assert abs(fused[0].score - 2 / 61) < 1e-10
