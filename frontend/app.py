"""Streamlit modern chat UI for the Hybrid RAG Engine with real-time SSE streaming."""

from __future__ import annotations

import json
import os
import time
from typing import Generator

import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MODES = {
    "⚡ Hybrid + Reranker (Recommended)": "hybrid_reranked",
    "🔀 Hybrid (Dense + BM25 Fusion)": "hybrid",
    "🎯 Naive (Vector Search Only)": "naive",
}

MODE_DESCRIPTIONS = {
    "hybrid_reranked": "Dense (BGE) + Sparse (BM25) fused via RRF (k=60), then reranked by Cross-Encoder (MiniLM). Highest precision.",
    "hybrid": "Combines dense semantic search and sparse lexical keyword search using Reciprocal Rank Fusion.",
    "naive": "Standard cosine similarity search over vector embeddings in Qdrant.",
}

PROMPT_SUGGESTIONS = [
    "🐳 What is Docker?",
    "⚡ FastAPI Dependency Injection",
    "💾 Volumes vs Bind Mounts",
    "📋 Task Tracker API Endpoints",
]

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Design System (CSS) ────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Gradient Header */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* Architecture Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 1.2rem;
    }

    .pipeline-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }

    .pipeline-badge.highlight {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
    }

    /* Chunk Cards */
    .chunk-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 10px;
        padding: 14px;
        margin: 10px 0;
        font-size: 0.86rem;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
    }

    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(51, 65, 85, 0.4);
    }

    .chunk-source {
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.8rem;
    }

    .method-pill {
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .method-reranked {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .method-dense {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .method-sparse {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Latency Badges */
    .timing-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(71, 85, 105, 0.4);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        color: #cbd5e1;
        margin: 2px 4px 2px 0;
    }

    .timing-badge b {
        color: #38bdf8;
    }

    /* Benchmark card in sidebar */
    .benchmark-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 10px;
        padding: 12px;
        margin: 10px 0;
    }

    .benchmark-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #c7d2fe;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }

    .benchmark-stat {
        font-size: 1.3rem;
        font-weight: 800;
        color: #34d399;
    }

    .benchmark-caption {
        font-size: 0.72rem;
        color: #94a3b8;
    }

    /* Pulse dot */
    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_timings" not in st.session_state:
    st.session_state.last_timings = {}

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Retrieval Configuration")

    selected_mode_key = st.selectbox(
        "Search Strategy",
        options=list(MODES.keys()),
        index=0,
        help="Select how documents are retrieved and scored.",
    )
    current_mode = MODES[selected_mode_key]

    st.caption(f"ℹ️ {MODE_DESCRIPTIONS[current_mode]}")

    top_k = st.slider(
        "Top Chunks to LLM (k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of most relevant chunks passed into the prompt context.",
    )

    st.divider()

    # System Telemetry & Health
    st.markdown("### 📊 System Health")
    try:
        health_resp = httpx.get(f"{API_URL}/health", timeout=4)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            if health_data.get("qdrant_connected"):
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">'
                    f'<span class="status-dot"></span> '
                    f'<span style="font-weight:600; font-size:0.85rem; color:#f1f5f9;">'
                    f'Engine Online ({health_data.get("chunk_count", 0)} chunks indexed)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("⚠️ Qdrant disconnected — start Docker container")
        else:
            st.error(f"API Degraded (HTTP {health_resp.status_code})")
    except Exception:
        st.error("⚠️ Backend API Unreachable at " + API_URL)

    # Benchmark Card
    st.markdown(
        """
        <div class="benchmark-card">
            <div class="benchmark-title">Ragas Evaluation Benchmark</div>
            <div class="benchmark-stat">+12.63%</div>
            <div class="benchmark-caption">Context Precision improvement with Hybrid RRF + Cross-Encoder reranking (83.08% → 95.71%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Document Management
    st.markdown("### 📥 Corpus Management")
    if st.button("🔄 Re-Index Documents (`data/`)", use_container_width=True):
        with st.spinner("Chunking and generating embeddings in Qdrant + BM25..."):
            try:
                resp = httpx.post(f"{API_URL}/ingest", timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✅ Indexed {data['num_documents']} docs → {data['num_chunks']} chunks!"
                    )
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_timings = {}
        st.session_state.last_chunks = []
        st.rerun()


# ── Main Header ───────────────────────────────────────────────
st.markdown('<div class="hero-title">⚡ Hybrid RAG Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Production-grade hybrid information retrieval combining <b>Dense Semantic Embeddings</b> '
    '(Qdrant + BGE), <b>Sparse Keyword Search</b> (BM25), <b>Reciprocal Rank Fusion (RRF)</b>, '
    'and <b>Cross-Encoder Reranking</b> with ultra-fast Groq LLM generation.'
    '</div>',
    unsafe_allow_html=True,
)

# Pipeline Badges
st.markdown(
    f"""
    <div class="badge-container">
        <span class="pipeline-badge highlight">Active Mode: <b>{selected_mode_key.split('(')[0].strip()}</b></span>
        <span class="pipeline-badge">🔷 Vector: BAAI/bge-small-en-v1.5</span>
        <span class="pipeline-badge">🔶 Sparse: BM25Okapi</span>
        <span class="pipeline-badge">🔀 Fusion: RRF (k=60)</span>
        <span class="pipeline-badge">🎯 Reranker: ms-marco-MiniLM</span>
        <span class="pipeline-badge">⚡ LLM: Groq Llama/GPT-OSS</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Prompt Suggestion Chips ───────────────────────────────────
st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#64748b; margin-bottom:6px;'>💡 QUICK START QUERIES</p>", unsafe_allow_html=True)
cols = st.columns(len(PROMPT_SUGGESTIONS))
for idx, suggestion in enumerate(PROMPT_SUGGESTIONS):
    with cols[idx]:
        if st.button(suggestion, key=f"sugg_{idx}", use_container_width=True):
            st.session_state.pending_prompt = suggestion

# ── Display Chat History ──────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander(f"📚 Retrieved Context Chunks ({len(msg['chunks'])})", expanded=False):
                for i, sc in enumerate(msg["chunks"], 1):
                    chunk = sc.get("chunk", {})
                    source = chunk.get("source", "unknown")
                    filename = source.rsplit("/", 1)[-1] if "/" in source else source
                    score = sc.get("score", 0.0)
                    method = sc.get("source_method", "reranked")
                    text = chunk.get("text", "")
                    tokens = chunk.get("token_count", 0)
                    chunk_idx = chunk.get("chunk_index", 0)

                    method_class = (
                        "method-reranked"
                        if "rerank" in method.lower()
                        else "method-dense"
                        if "dense" in method.lower()
                        else "method-sparse"
                    )

                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <div class="chunk-header">
                                <span class="chunk-source">📄 {filename} <span style="color:#64748b; font-weight:400;">(Chunk #{chunk_idx}, {tokens} tokens)</span></span>
                                <span class="method-pill {method_class}">{method} • Score: {score:.4f}</span>
                            </div>
                            <div style="color:#e2e8f0; font-size:0.84rem;">{text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        if msg.get("timings"):
            timings_html = "".join(
                f'<span class="timing-badge">{k}: <b>{v*1000:.1f}ms</b></span>'
                for k, v in msg["timings"].items()
            )
            st.markdown(f"<div style='margin-top:8px;'>{timings_html}</div>", unsafe_allow_html=True)


# ── Query Streamer Function ───────────────────────────────────
def stream_response_from_api(
    question: str, mode: str, top_k_val: int
) -> Generator[tuple[str, list, dict], None, None]:
    """
    Stream tokens from FastAPI SSE /query/stream endpoint.
    Yields (token_chunk, chunks_data, timings_data).
    """
    retrieved_chunks = []
    timings = {}

    url = f"{API_URL}/query/stream"
    params = {"question": question, "mode": mode, "top_k": top_k_val}

    with httpx.stream("GET", url, params=params, timeout=60.0) as response:
        if response.status_code != 200:
            yield f"⚠️ API Error ({response.status_code}): {response.read().decode('utf-8')}", [], {}
            return

        current_event = None
        for line in response.iter_lines():
            if not line:
                continue

            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: "):
                raw_data = line[6:].strip()

                if current_event == "chunks":
                    try:
                        retrieved_chunks = json.loads(raw_data)
                    except Exception:
                        retrieved_chunks = []
                elif current_event == "token":
                    yield raw_data, retrieved_chunks, timings
                elif current_event == "done":
                    try:
                        timings = json.loads(raw_data)
                    except Exception:
                        timings = {}
                    yield "", retrieved_chunks, timings


# ── Handle Input Submission ───────────────────────────────────
prompt_input = st.chat_input("Ask a question about your indexed documents...")
user_prompt = prompt_input or st.session_state.pending_prompt

if user_prompt:
    st.session_state.pending_prompt = None

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # Stream assistant answer
    with st.chat_message("assistant", avatar="⚡"):
        response_placeholder = st.empty()
        full_response = ""
        final_chunks = []
        final_timings = {}

        try:
            for token, chunks, timings in stream_response_from_api(
                user_prompt, current_mode, top_k
            ):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
                if chunks:
                    final_chunks = chunks
                if timings:
                    final_timings = timings

            # Render final static response without blinking cursor
            response_placeholder.markdown(full_response)

            # Display source chunks dropdown
            if final_chunks:
                with st.expander(
                    f"📚 Retrieved Context Chunks ({len(final_chunks)})",
                    expanded=False,
                ):
                    for i, sc in enumerate(final_chunks, 1):
                        chunk = sc.get("chunk", {})
                        source = chunk.get("source", "unknown")
                        filename = (
                            source.rsplit("/", 1)[-1] if "/" in source else source
                        )
                        score = sc.get("score", 0.0)
                        method = sc.get("source_method", "reranked")
                        text = chunk.get("text", "")
                        tokens = chunk.get("token_count", 0)
                        chunk_idx = chunk.get("chunk_index", 0)

                        method_class = (
                            "method-reranked"
                            if "rerank" in method.lower()
                            else "method-dense"
                            if "dense" in method.lower()
                            else "method-sparse"
                        )

                        st.markdown(
                            f"""
                            <div class="chunk-card">
                                <div class="chunk-header">
                                    <span class="chunk-source">📄 {filename} <span style="color:#64748b; font-weight:400;">(Chunk #{chunk_idx}, {tokens} tokens)</span></span>
                                    <span class="method-pill {method_class}">{method} • Score: {score:.4f}</span>
                                </div>
                                <div style="color:#e2e8f0; font-size:0.84rem;">{text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Display timing metrics badges
            if final_timings:
                timings_html = "".join(
                    f'<span class="timing-badge">{k}: <b>{v*1000:.1f}ms</b></span>'
                    for k, v in final_timings.items()
                )
                st.markdown(
                    f"<div style='margin-top:8px;'>{timings_html}</div>",
                    unsafe_allow_html=True,
                )

            # Save to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "chunks": final_chunks,
                    "timings": final_timings,
                }
            )

        except Exception as e:
            st.error(f"Connection Error: {e}")
