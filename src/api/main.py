"""FastAPI application entry point with lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import app_state
from src.api.routes import health, ingest, query
from src.observability.tracing import init_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, connect services, init tracing. Shutdown: cleanup."""
    logger.info("🚀 Starting Hybrid RAG Engine...")

    # Initialize observability (non-blocking, degrades gracefully)
    init_tracing()

    # Load all models and connect to Qdrant
    app_state.startup()

    logger.info("✅ Hybrid RAG Engine ready")
    yield

    # Shutdown
    logger.info("Shutting down...")
    app_state.shutdown()


app = FastAPI(
    title="Hybrid RAG Engine",
    description=(
        "A hybrid retrieval-augmented generation system combining dense (Qdrant) "
        "and sparse (BM25) search with RRF fusion, cross-encoder reranking, "
        "and LLM generation via Groq."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])
