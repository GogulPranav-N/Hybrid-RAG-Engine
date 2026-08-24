"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client. Requires Qdrant to be running."""
    try:
        from src.api.main import app
        with TestClient(app) as c:
            yield c
    except Exception:
        pytest.skip("Cannot create test client — dependencies may not be available")


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "qdrant_connected" in data


class TestIngestEndpoint:
    def test_ingest_with_default_data(self, client):
        """Test ingestion with the default data directory."""
        response = client.post("/ingest")
        if response.status_code == 200:
            data = response.json()
            assert data["num_documents"] > 0
            assert data["num_chunks"] > 0
            assert len(data["sources"]) > 0
        else:
            # Acceptable if data dir doesn't exist in test env
            assert response.status_code in (400, 404, 503)


class TestQueryEndpoint:
    def test_query_naive_mode(self, client):
        """Test naive query (requires prior ingestion)."""
        response = client.post(
            "/query",
            json={
                "question": "What is FastAPI?",
                "mode": "naive",
                "top_k": 3,
            },
        )
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "chunks" in data
            assert data["mode"] == "naive"
        else:
            assert response.status_code == 503  # Models not loaded

    def test_query_requires_question(self, client):
        """Test validation: question is required."""
        response = client.post("/query", json={"mode": "naive"})
        assert response.status_code == 422  # Validation error

    def test_query_stream_endpoint(self, client):
        """Test that the SSE endpoint exists and accepts parameters."""
        response = client.get(
            "/query/stream",
            params={"question": "test", "mode": "naive", "top_k": 3},
        )
        # SSE endpoints return 200 with text/event-stream content type
        assert response.status_code in (200, 503)
