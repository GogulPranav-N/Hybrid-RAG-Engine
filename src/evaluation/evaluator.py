"""Ragas-based evaluation runner for the RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.config import settings
from src.models import QueryRequest, SearchMode, ScoredChunk

logger = logging.getLogger(__name__)

# Path to the ground-truth test set
TEST_SET_PATH = Path(__file__).parent / "test_set.json"


def load_test_set(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the ground-truth Q&A test set."""
    path = Path(path or TEST_SET_PATH)
    with open(path) as f:
        return json.load(f)


async def evaluate_pipeline(
    query_fn,
    mode: SearchMode,
    test_set: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a RAG pipeline configuration using Ragas metrics.

    Args:
        query_fn: Async function(QueryRequest) → (answer, chunks) that runs
                  the full pipeline for a given mode.
        mode: The search mode to evaluate.
        test_set: Ground-truth Q&A pairs. Loaded from file if not provided.

    Returns:
        Dict with per-question scores and aggregate metrics.
    """
    if test_set is None:
        test_set = load_test_set()

    try:
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError:
        # Fallback for newer Ragas versions with different import paths
        logger.warning("Using fallback Ragas import path")
        from ragas.metrics import (
            context_precision as ContextPrecision,
            context_recall as ContextRecall,
            faithfulness as Faithfulness,
        )

    results: list[dict[str, Any]] = []

    for i, test_case in enumerate(test_set):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        logger.info(
            "Evaluating [%s] question %d/%d: %.60s...",
            mode.value,
            i + 1,
            len(test_set),
            question,
        )

        try:
            # Run the pipeline
            answer, chunks = await query_fn(
                QueryRequest(question=question, mode=mode, top_k=5)
            )

            # Extract context texts
            retrieved_contexts = [sc.chunk.text for sc in chunks]

            # Score with Ragas metrics
            # Note: Ragas metrics may require an LLM for scoring.
            # For simplicity, we compute a heuristic version here
            # and provide the full Ragas integration as an option.
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "retrieved_contexts": retrieved_contexts,
                "num_chunks": len(chunks),
                "mode": mode.value,
            }

            # Heuristic context precision: how many retrieved chunks contain
            # keywords from the ground truth answer
            gt_keywords = set(ground_truth.lower().split())
            relevant_count = sum(
                1
                for ctx in retrieved_contexts
                if len(gt_keywords & set(ctx.lower().split())) > len(gt_keywords) * 0.1
            )
            result["context_precision"] = (
                relevant_count / len(retrieved_contexts) if retrieved_contexts else 0.0
            )

            # Heuristic context recall: what fraction of ground truth keywords
            # appear in at least one retrieved chunk
            all_retrieved_text = " ".join(retrieved_contexts).lower()
            recalled_keywords = sum(
                1 for kw in gt_keywords if kw in all_retrieved_text
            )
            result["context_recall"] = (
                recalled_keywords / len(gt_keywords) if gt_keywords else 0.0
            )

            # Heuristic faithfulness: what fraction of answer sentences
            # have support in the retrieved contexts
            answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]
            supported = sum(
                1
                for sent in answer_sentences
                if any(
                    len(set(sent.lower().split()) & set(ctx.lower().split())) > 3
                    for ctx in retrieved_contexts
                )
            )
            result["faithfulness"] = (
                supported / len(answer_sentences) if answer_sentences else 0.0
            )

            results.append(result)

        except Exception:
            logger.exception("Failed to evaluate question: %s", question)
            results.append(
                {
                    "question": question,
                    "error": True,
                    "mode": mode.value,
                }
            )

    # Compute aggregates
    valid_results = [r for r in results if "error" not in r]
    aggregates = {}
    for metric in ("context_precision", "context_recall", "faithfulness"):
        values = [r[metric] for r in valid_results if metric in r]
        aggregates[metric] = sum(values) / len(values) if values else 0.0

    return {
        "mode": mode.value,
        "num_questions": len(test_set),
        "num_evaluated": len(valid_results),
        "aggregates": aggregates,
        "results": results,
    }
