"""Compare all three RAG configurations and produce a results table."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import evaluate_pipeline, load_test_set
from src.models import SearchMode

logger = logging.getLogger(__name__)


async def run_comparison(
    query_fn,
    output_path: str | Path = "evaluation_results.json",
) -> dict[str, Any]:
    """
    Run the full evaluation across all three modes and produce a comparison.

    Args:
        query_fn: Async function(QueryRequest) → (answer, chunks).
        output_path: Where to save the full results JSON.

    Returns:
        Dict with comparison table data.
    """
    test_set = load_test_set()
    modes = [SearchMode.NAIVE, SearchMode.HYBRID, SearchMode.HYBRID_RERANKED]

    all_results: dict[str, Any] = {}

    for mode in modes:
        logger.info("=" * 60)
        logger.info("Evaluating mode: %s", mode.value)
        logger.info("=" * 60)

        result = await evaluate_pipeline(query_fn, mode, test_set)
        all_results[mode.value] = result

    # Build comparison table
    comparison = {
        "modes": {},
        "table_markdown": _format_table(all_results),
    }

    for mode_name, result in all_results.items():
        comparison["modes"][mode_name] = result["aggregates"]

    # Save full results
    output_path = Path(output_path)
    full_output = {
        "comparison": comparison,
        "detailed_results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(full_output, f, indent=2)

    logger.info("Results saved to %s", output_path)
    print("\n" + comparison["table_markdown"])

    return comparison


def _format_table(all_results: dict[str, Any]) -> str:
    """Format comparison results as a markdown table."""
    lines = [
        "| Configuration | Context Precision | Context Recall | Faithfulness |",
        "|---|---|---|---|",
    ]

    display_names = {
        "naive": "Naive (vector only)",
        "hybrid": "Hybrid (+ BM25/RRF)",
        "hybrid_reranked": "Hybrid + Reranker",
    }

    for mode_name, result in all_results.items():
        agg = result["aggregates"]
        lines.append(
            f"| {display_names.get(mode_name, mode_name)} "
            f"| {agg.get('context_precision', 0):.2%} "
            f"| {agg.get('context_recall', 0):.2%} "
            f"| {agg.get('faithfulness', 0):.2%} |"
        )

    return "\n".join(lines)
