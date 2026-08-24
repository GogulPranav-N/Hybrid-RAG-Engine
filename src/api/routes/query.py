"""Query endpoint with naive, hybrid, and hybrid+reranked modes + SSE streaming."""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from sse_starlette import EventSourceResponse

from src.api.dependencies import app_state
from src.generation.llm import generate, generate_stream
from src.models import QueryRequest, QueryResponse, SearchMode, ScoredChunk
from src.observability.tracing import traced_span
from src.retrieval.dense import dense_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.retrieval.sparse import sparse_search

router = APIRouter()
logger = logging.getLogger(__name__)


async def _retrieve(request: QueryRequest) -> tuple[list[ScoredChunk], dict[str, float]]:
    """
    Run the retrieval pipeline based on the requested mode.

    Returns (scored_chunks, timings_dict).
    """
    timings: dict[str, float] = {}

    if not app_state.embedding_model or not app_state.qdrant_client:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # ── Step 1: Dense search (always) ─────────────────────────
    with traced_span("dense_search", {"query": request.question}) as span:
        t0 = time.perf_counter()
        dense_results = dense_search(
            query=request.question,
            embedding_model=app_state.embedding_model,
            qdrant_client=app_state.qdrant_client,
            top_k=request.top_k if request.mode == SearchMode.NAIVE else 20,
        )
        timings["dense_search"] = time.perf_counter() - t0
        span["result_count"] = len(dense_results)

    # ── Naive mode: return dense results directly ─────────────
    if request.mode == SearchMode.NAIVE:
        return dense_results[: request.top_k], timings

    # ── Step 2: Sparse search (hybrid modes) ──────────────────
    if not app_state.bm25_index:
        raise HTTPException(
            status_code=503,
            detail="BM25 index not available — run ingestion first",
        )

    with traced_span("sparse_search", {"query": request.question}) as span:
        t0 = time.perf_counter()
        sparse_results = sparse_search(
            query=request.question,
            bm25_index=app_state.bm25_index,
            top_k=20,
        )
        timings["sparse_search"] = time.perf_counter() - t0
        span["result_count"] = len(sparse_results)

    # ── Step 3: Reciprocal Rank Fusion ────────────────────────
    with traced_span("rrf_fusion") as span:
        t0 = time.perf_counter()
        fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
        timings["fusion"] = time.perf_counter() - t0
        span["result_count"] = len(fused_results)

    # ── Hybrid mode (no reranking): return fused results ──────
    if request.mode == SearchMode.HYBRID:
        return fused_results[: request.top_k], timings

    # ── Step 4: Cross-encoder reranking ───────────────────────
    if not app_state.cross_encoder:
        raise HTTPException(status_code=503, detail="Cross-encoder not loaded")

    with traced_span("reranking") as span:
        t0 = time.perf_counter()
        # Rerank top-25 fused results, keep top_k
        reranked_results = rerank(
            query=request.question,
            scored_chunks=fused_results[:25],
            cross_encoder=app_state.cross_encoder,
            top_k=request.top_k,
        )
        timings["reranking"] = time.perf_counter() - t0
        span["result_count"] = len(reranked_results)

    return reranked_results, timings


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system.

    Modes:
    - **naive**: Dense vector search only
    - **hybrid**: Dense + BM25 with RRF fusion
    - **hybrid_reranked**: Dense + BM25 + RRF + cross-encoder reranking
    """
    # Retrieve
    chunks, timings = await _retrieve(request)

    if not chunks:
        return QueryResponse(
            answer="I couldn't find any relevant information to answer your question.",
            chunks=[],
            mode=request.mode,
            timings=timings,
        )

    # Generate
    with traced_span("generation", {"mode": request.mode.value}) as span:
        t0 = time.perf_counter()
        answer = generate(request.question, chunks)
        timings["generation"] = time.perf_counter() - t0
        span["answer_length"] = len(answer)

    return QueryResponse(
        answer=answer,
        chunks=chunks,
        mode=request.mode,
        timings=timings,
    )


@router.get("/stream")
async def query_stream(
    request: Request,
    question: str,
    mode: SearchMode = SearchMode.HYBRID_RERANKED,
    top_k: int = 5,
):
    """
    Stream the RAG answer via Server-Sent Events (SSE).

    First sends retrieved chunks as a 'chunks' event, then streams
    answer tokens as 'token' events, and finally sends timings as
    a 'done' event.
    """
    query_request = QueryRequest(question=question, mode=mode, top_k=top_k)

    # Retrieve (non-streaming)
    chunks, timings = await _retrieve(query_request)

    async def event_generator():
        # 1. Send retrieved chunks
        chunks_data = [sc.model_dump() for sc in chunks]
        yield {"event": "chunks", "data": json.dumps(chunks_data)}

        if not chunks:
            yield {
                "event": "token",
                "data": "I couldn't find any relevant information to answer your question.",
            }
            yield {"event": "done", "data": json.dumps(timings)}
            return

        # 2. Stream answer tokens
        t0 = time.perf_counter()
        async for token in generate_stream(question, chunks):
            if await request.is_disconnected():
                return
            yield {"event": "token", "data": token}

        timings["generation"] = time.perf_counter() - t0

        # 3. Send completion signal with timings
        yield {"event": "done", "data": json.dumps(timings)}

    return EventSourceResponse(event_generator())
