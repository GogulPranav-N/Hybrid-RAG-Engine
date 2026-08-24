"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import app_state
from src.config import settings
from src.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check system health: Qdrant connection + collection status."""
    qdrant_connected = False
    collection_exists = False
    chunk_count = 0

    try:
        if app_state.qdrant_client:
            # Check Qdrant connectivity
            collections = app_state.qdrant_client.get_collections().collections
            qdrant_connected = True

            # Check if our collection exists
            collection_names = [c.name for c in collections]
            if settings.collection_name in collection_names:
                collection_exists = True
                info = app_state.qdrant_client.get_collection(settings.collection_name)
                chunk_count = info.points_count or 0
    except Exception:
        pass

    return HealthResponse(
        status="ok" if qdrant_connected else "degraded",
        qdrant_connected=qdrant_connected,
        collection_exists=collection_exists,
        chunk_count=chunk_count,
    )
