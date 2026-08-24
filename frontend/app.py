"""Streamlit chat UI for the Hybrid RAG Engine."""

from __future__ import annotations

import json
import os
import time

import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
MODES = {
    "Naive (Vector Only)": "naive",
    "Hybrid (Vector + BM25)": "hybrid",
    "Hybrid + Reranker": "hybrid_reranked",
}

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .chunk-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.85rem;
    }
    .chunk-source {
        color: #89b4fa;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .chunk-score {
        color: #a6e3a1;
        font-size: 0.8rem;
    }
    .timing-metric {
        display: inline-block;
        background: #313244;
        padding: 4px 10px;
        border-radius: 12px;
        margin: 2px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode_label = st.selectbox(
        "Search Mode",
        options=list(MODES.keys()),
        index=2,  # Default to hybrid+reranked
        help="Select the retrieval strategy",
    )
    mode = MODES[mode_label]

    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=15,
        value=5,
        help="Number of chunks to retrieve",
    )

    st.divider()

    # Health check
    st.markdown("### 📊 System Status")
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        if health.get("qdrant_connected"):
            st.success(f"Qdrant: Connected ({health.get('chunk_count', 0)} chunks)")
        else:
            st.error("Qdrant: Disconnected")
    except Exception:
        st.error("API: Unreachable")

    st.divider()

    # Ingestion trigger
    st.markdown("### 📥 Data Ingestion")
    if st.button("🔄 Re-ingest Documents", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            try:
                resp = httpx.post(f"{API_URL}/ingest", timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Ingested {data['num_documents']} docs → "
                        f"{data['num_chunks']} chunks"
                    )
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Timings display
    if "last_timings" in st.session_state and st.session_state.last_timings:
        st.markdown("### ⏱️ Last Query Timings")
        for step, elapsed in st.session_state.last_timings.items():
            st.markdown(
                f'<span class="timing-metric">{step}: {elapsed:.3f}s</span>',
                unsafe_allow_html=True,
            )

# ── Main Content ──────────────────────────────────────────────
st.markdown('<p class="main-header">🔍 Hybrid RAG Engine</p>', unsafe_allow_html=True)
st.markdown(
    "Ask questions about FastAPI, Docker, or the Task Tracker API. "
    f"Using **{mode_label}** mode."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_timings" not in st.session_state:
    st.session_state.last_timings = {}

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the API
    with st.chat_message("assistant"):
        try:
            # Use the non-streaming endpoint for simplicity
            response = httpx.post(
                f"{API_URL}/query",
                json={
                    "question": prompt,
                    "mode": mode,
                    "top_k": top_k,
                },
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                chunks = data.get("chunks", [])
                timings = data.get("timings", {})

                # Display answer
                st.markdown(answer)

                # Save state
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.last_timings = timings
                st.session_state.last_chunks = chunks

                # Display source chunks in an expander
                if chunks:
                    with st.expander(f"📚 Source Chunks ({len(chunks)})", expanded=False):
                        for i, sc in enumerate(chunks, 1):
                            chunk = sc.get("chunk", {})
                            source = chunk.get("source", "unknown")
                            filename = (
                                source.rsplit("/", 1)[-1] if "/" in source else source
                            )
                            score = sc.get("score", 0)
                            method = sc.get("source_method", "")
                            text = chunk.get("text", "")

                            st.markdown(
                                f'<div class="chunk-card">'
                                f'<span class="chunk-source">📄 {filename}</span> · '
                                f'<span class="chunk-score">Score: {score:.4f} ({method})</span>'
                                f"<br><br>{text[:300]}{'...' if len(text) > 300 else ''}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
            else:
                st.error(f"API error: {response.text}")

        except httpx.TimeoutException:
            st.error("Request timed out. Is the API running?")
        except httpx.ConnectError:
            st.error(
                f"Cannot connect to API at {API_URL}. "
                "Make sure the server is running."
            )
        except Exception as e:
            st.error(f"Error: {e}")
