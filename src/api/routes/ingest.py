"""Document ingestion endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.dependencies import app_state
from src.config import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import run_indexing
from src.ingestion.loader import load_directory
from src.models import IngestRequest, IngestResponse
from src.retrieval.sparse import BM25Index

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest | None = None) -> IngestResponse:
    """
    Load, chunk, and index all documents from the data directory.

    This replaces any existing index (full re-ingestion).
    """
    data_dir = (request and request.data_dir) or str(settings.data_dir)
    chunk_size = (request and request.chunk_size) or settings.chunk_size
    chunk_overlap = (request and request.chunk_overlap) or settings.chunk_overlap

    if not app_state.embedding_model or not app_state.qdrant_client:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        # 1. Load documents
        documents = load_directory(data_dir)
        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No supported documents found in {data_dir}",
            )

        # 2. Chunk
        chunks = chunk_documents(documents, chunk_size, chunk_overlap)

        # 3. Index (Qdrant + BM25)
        run_indexing(chunks, app_state.embedding_model, app_state.qdrant_client)

        # 4. Reload BM25 index into memory
        app_state.bm25_index = BM25Index.load()

        sources = list({doc.source_path for doc in documents})

        logger.info(
            "Ingestion complete: %d docs → %d chunks from %d sources",
            len(documents),
            len(chunks),
            len(sources),
        )

        return IngestResponse(
            num_documents=len(documents),
            num_chunks=len(chunks),
            sources=sources,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
