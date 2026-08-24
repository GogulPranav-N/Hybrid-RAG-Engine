"""Pydantic schemas for documents, chunks, API requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────


class SearchMode(str, Enum):
    """Retrieval strategy selector."""

    NAIVE = "naive"
    HYBRID = "hybrid"
    HYBRID_RERANKED = "hybrid_reranked"


class FileType(str, Enum):
    """Supported document file types."""

    MARKDOWN = "markdown"
    PDF = "pdf"
    PYTHON = "python"
    TEXT = "text"


# ── Domain Models ─────────────────────────────────────────────


class Document(BaseModel):
    """A raw loaded document before chunking."""

    text: str
    source_path: str
    file_type: FileType
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk with provenance metadata."""

    id: str
    text: str
    source: str
    chunk_index: int
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A chunk annotated with a retrieval or reranking score."""

    chunk: Chunk
    score: float
    source_method: str = ""  # "dense", "sparse", "fused", "reranked"


# ── API Schemas ───────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Incoming query from the user."""

    question: str = Field(..., min_length=1, max_length=2000)
    mode: SearchMode = SearchMode.HYBRID_RERANKED
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Full query response with answer, sources, and timing breakdown."""

    answer: str
    chunks: list[ScoredChunk]
    mode: SearchMode
    timings: dict[str, float] = Field(default_factory=dict)  # step → seconds


class IngestRequest(BaseModel):
    """Optional overrides for ingestion."""

    data_dir: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class IngestResponse(BaseModel):
    """Result of an ingestion run."""

    num_documents: int
    num_chunks: int
    sources: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    qdrant_connected: bool = False
    collection_exists: bool = False
    chunk_count: int = 0
