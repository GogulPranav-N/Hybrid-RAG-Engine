"""Application configuration via pydantic-settings, loaded from .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — reads from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "hybrid_rag"

    # ── Models ────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "openai/gpt-oss-120b"

    # ── LLM Provider ─────────────────────────────────────────
    groq_api_key: str = ""

    # ── Chunking ──────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 50

    # ── Retrieval ─────────────────────────────────────────────
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    # ── Observability ─────────────────────────────────────────
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "hybrid-rag-engine"

    # ── Data ──────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    bm25_index_path: Path = Path("./data/bm25_index.pkl")


# Singleton — import this everywhere
settings = Settings()
