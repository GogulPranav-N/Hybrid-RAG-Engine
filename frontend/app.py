"""Enterprise Hybrid RAG Control Console & Query Workbench."""

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
    "Hybrid + Cross-Encoder Reranker (Default)": "hybrid_reranked",
    "Hybrid (Dense Vector + BM25 Fusion)": "hybrid",
    "Dense Semantic Search Only (Naive)": "naive",
}

MODE_DESCRIPTIONS = {
    "hybrid_reranked": "Dense embeddings (BGE-Small) + Lexical sparse (BM25) fused via RRF (k=60), reranked by Cross-Encoder (MiniLM-L6).",
    "hybrid": "Reciprocal Rank Fusion over dense vector cosine similarity and sparse BM25 scores.",
    "naive": "Single-stage cosine similarity search over dense vector embeddings in Qdrant.",
}

PROMPT_SUGGESTIONS = [
    ("Docker Architecture", "What is Docker and how do containers differ from virtual machines?"),
    ("FastAPI Dependencies", "Explain FastAPI dependency injection pattern with code examples."),
    ("Storage Volumes", "What is the difference between Docker volumes and bind mounts?"),
    ("Task Tracker Schema", "List all endpoints and data schemas in the Task Tracker API."),
]

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG Engine Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Enterprise Styling ────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global typography */
    body, .stMarkdown, p {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Background */
    .stApp {
        background-color: #0b0e14;
    }

    /* Top Navigation Header */
    .header-panel {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 22px;
        background: #111620;
        border: 1px solid #1e2638;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.01em;
        text-transform: uppercase;
    }

    .brand-sub {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 400;
        margin-top: 2px;
    }

    .tech-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-left: 6px;
    }

    .tech-pill-primary {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38bdf8;
    }

    .tech-pill-secondary {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
    }

    /* Custom Chat Message Cards */
    .msg-box-user {
        background: #131926;
        border: 1px solid #1f293d;
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 14px 16px;
        margin: 12px 0;
    }

    .msg-box-assistant {
        background: #0f141d;
        border: 1px solid #1b2333;
        border-left: 3px solid #10b981;
        border-radius: 6px;
        padding: 16px 18px;
        margin: 12px 0;
    }

    .msg-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .msg-role-user {
        font-size: 0.75rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .msg-role-assistant {
        font-size: 0.75rem;
        font-weight: 700;
        color: #10b981;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .msg-content {
        color: #e2e8f0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Metric Cards */
    .metric-card {
        background: #111620;
        border: 1px solid #1e2638;
        border-radius: 6px;
        padding: 14px;
        text-align: left;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-value-accent {
        color: #10b981;
    }

    /* Chunk Cards */
    .chunk-container {
        background: #0d1118;
        border: 1px solid #1a2232;
        border-radius: 6px;
        padding: 12px 14px;
        margin: 8px 0;
    }

    .chunk-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 1px solid #161d2b;
    }

    .chunk-filename {
        color: #38bdf8;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .chunk-tag {
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .chunk-tag-rerank {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .chunk-tag-dense {
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    .chunk-tag-sparse {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .chunk-body {
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* Latency Badges */
    .timing-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #111620;
        border: 1px solid #1e2638;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        color: #64748b;
        margin-right: 6px;
        margin-top: 6px;
    }

    .timing-badge b {
        color: #e2e8f0;
    }

    /* Status Indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #10b981;
    }

    .status-dot {
        height: 6px;
        width: 6px;
        background-color: #10b981;
        border-radius: 50%;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #080b10 !important;
        border-right: 1px solid #161c28 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ── Header Panel ──────────────────────────────────────────────
st.markdown(
    """
    <div class="header-panel">
        <div>
            <div class="brand-title">HYBRID RAG CONTROL CONSOLE</div>
            <div class="brand-sub">Production Hybrid Search Pipeline • Dense Embedding + BM25 Sparse + RRF + Cross-Encoder</div>
        </div>
        <div>
            <span class="tech-pill tech-pill-primary">Qdrant Vector DB</span>
            <span class="tech-pill tech-pill-secondary">BM25Okapi</span>
            <span class="tech-pill tech-pill-primary">MiniLM Reranker</span>
            <span class="tech-pill tech-pill-secondary">Groq LLM</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar Controls ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### Pipeline Configuration")

    selected_mode_label = st.selectbox(
        "Retrieval Strategy",
        options=list(MODES.keys()),
        index=0,
        help="Select the retrieval and ranking strategy.",
    )
    current_mode = MODES[selected_mode_label]

    st.caption(MODE_DESCRIPTIONS[current_mode])

    top_k = st.slider(
        "Top Chunks (k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of retrieved chunks supplied in prompt context.",
    )

    st.divider()

    # System Status
    st.markdown("### Infrastructure Telemetry")
    try:
        health_resp = httpx.get(f"{API_URL}/health", timeout=3.0)
        if health_resp.status_code == 200:
            hdata = health_resp.json()
            chunks_cnt = hdata.get("chunk_count", 0)
            st.markdown(
                f"""
                <div style="background:#111620; border:1px solid #1e2638; border-radius:6px; padding:10px; margin-bottom:12px;">
                    <div class="status-badge">
                        <span class="status-dot"></span> SYSTEM OPERATIONAL
                    </div>
                    <div style="font-size:0.75rem; color:#64748b; margin-top:6px; line-height:1.4;">
                        Qdrant Status: <b>Connected</b><br>
                        Indexed Vectors: <b>{chunks_cnt} chunks</b><br>
                        BM25 Inverted Index: <b>Loaded</b><br>
                        Execution: <b>Non-blocking Async</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Degraded Status: HTTP {health_resp.status_code}")
    except Exception:
        st.error(f"API Unreachable at {API_URL}")

    st.divider()

    # Corpus Operations
    st.markdown("### Corpus Operations")
    if st.button("Re-Index Corpus (`data/`)", use_container_width=True):
        with st.spinner("Processing documents into Qdrant & BM25..."):
            try:
                resp = httpx.post(f"{API_URL}/ingest", timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Indexed {data['num_documents']} documents into {data['num_chunks']} chunks.")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    if st.button("Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Tabs ─────────────────────────────────────────────────
tab_chat, tab_metrics, tab_corpus = st.tabs([
    "Query Console",
    "Evaluation Benchmarks",
    "Indexed Documents",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: Query Console
# ═══════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("<div style='font-size:0.72rem; font-weight:600; color:#64748b; text-transform:uppercase; margin-bottom:6px;'>Sample Queries</div>", unsafe_allow_html=True)
    pcols = st.columns(len(PROMPT_SUGGESTIONS))
    for i, (label, text) in enumerate(PROMPT_SUGGESTIONS):
        with pcols[i]:
            if st.button(label, key=f"q_{i}", use_container_width=True):
                st.session_state.pending_prompt = text

    # Render Chat History with Clean Card Layout
    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        box_class = "msg-box-user" if is_user else "msg-box-assistant"
        role_class = "msg-role-user" if is_user else "msg-role-assistant"
        role_title = "USER QUERY" if is_user else "ENGINE RESPONSE"

        st.markdown(
            f"""
            <div class="{box_class}">
                <div class="msg-header">
                    <span class="{role_class}">{role_title}</span>
                </div>
                <div class="msg-content">{msg['content']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not is_user and msg.get("chunks"):
            with st.expander(f"Retrieved Context Chunks ({len(msg['chunks'])})", expanded=False):
                for idx, sc in enumerate(msg["chunks"], 1):
                    chunk = sc.get("chunk", {})
                    source = chunk.get("source", "doc")
                    filename = source.rsplit("/", 1)[-1]
                    score = sc.get("score", 0.0)
                    method = sc.get("source_method", "reranked")
                    text_body = chunk.get("text", "")
                    tokens = chunk.get("token_count", 0)
                    chunk_i = chunk.get("chunk_index", 0)

                    tag_class = (
                        "chunk-tag-rerank" if "rerank" in method.lower()
                        else "chunk-tag-dense" if "dense" in method.lower()
                        else "chunk-tag-sparse"
                    )

                    st.markdown(
                        f"""
                        <div class="chunk-container">
                            <div class="chunk-meta-row">
                                <span class="chunk-filename">{filename} (Chunk {chunk_i}, {tokens} tokens)</span>
                                <span class="chunk-tag {tag_class}">{method} | Score: {score:.4f}</span>
                            </div>
                            <div class="chunk-body">{text_body}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if not is_user and msg.get("timings"):
            thtml = "".join(
                f'<span class="timing-badge">{k}: <b>{v*1000:.1f}ms</b></span>'
                for k, v in msg["timings"].items()
            )
            st.markdown(f"<div style='margin-bottom:12px;'>{thtml}</div>", unsafe_allow_html=True)

    # Streaming API Function
    def stream_query(question: str, mode_val: str, k_val: int) -> Generator[tuple[str, list, dict], None, None]:
        url = f"{API_URL}/query/stream"
        params = {"question": question, "mode": mode_val, "top_k": k_val}
        chunks = []
        timings = {}

        with httpx.stream("GET", url, params=params, timeout=60.0) as resp:
            if resp.status_code != 200:
                yield f"Error ({resp.status_code}): {resp.read().decode('utf-8')}", [], {}
                return

            event = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    event = line[7:].strip()
                elif line.startswith("data: "):
                    payload = line[6:].strip()
                    if event == "chunks":
                        try:
                            chunks = json.loads(payload)
                        except Exception:
                            chunks = []
                    elif event == "token":
                        yield payload, chunks, timings
                    elif event == "done":
                        try:
                            timings = json.loads(payload)
                        except Exception:
                            timings = {}
                        yield "", chunks, timings

    # Chat Input Handler
    user_input = st.chat_input("Enter query regarding indexed technical documentation...")
    active_prompt = user_input or st.session_state.pending_prompt

    if active_prompt:
        st.session_state.pending_prompt = None

        # User Query Box
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        st.markdown(
            f"""
            <div class="msg-box-user">
                <div class="msg-header">
                    <span class="msg-role-user">USER QUERY</span>
                </div>
                <div class="msg-content">{active_prompt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Assistant Response Container
        assistant_card = st.empty()
        full_text = ""
        final_chunks = []
        final_timings = {}

        try:
            for token_piece, c_list, t_dict in stream_query(active_prompt, current_mode, top_k):
                full_text += token_piece
                assistant_card.markdown(
                    f"""
                    <div class="msg-box-assistant">
                        <div class="msg-header">
                            <span class="msg-role-assistant">ENGINE RESPONSE</span>
                        </div>
                        <div class="msg-content">{full_text}▌</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if c_list:
                    final_chunks = c_list
                if t_dict:
                    final_timings = t_dict

            # Final static render without cursor
            assistant_card.markdown(
                f"""
                <div class="msg-box-assistant">
                    <div class="msg-header">
                        <span class="msg-role-assistant">ENGINE RESPONSE</span>
                    </div>
                    <div class="msg-content">{full_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if final_chunks:
                with st.expander(f"Retrieved Context Chunks ({len(final_chunks)})", expanded=False):
                    for idx, sc in enumerate(final_chunks, 1):
                        chunk = sc.get("chunk", {})
                        source = chunk.get("source", "doc")
                        filename = source.rsplit("/", 1)[-1]
                        score = sc.get("score", 0.0)
                        method = sc.get("source_method", "reranked")
                        text_body = chunk.get("text", "")
                        tokens = chunk.get("token_count", 0)
                        chunk_i = chunk.get("chunk_index", 0)

                        tag_class = (
                            "chunk-tag-rerank" if "rerank" in method.lower()
                            else "chunk-tag-dense" if "dense" in method.lower()
                            else "chunk-tag-sparse"
                        )

                        st.markdown(
                            f"""
                            <div class="chunk-container">
                                <div class="chunk-meta-row">
                                    <span class="chunk-filename">{filename} (Chunk {chunk_i}, {tokens} tokens)</span>
                                    <span class="chunk-tag {tag_class}">{method} | Score: {score:.4f}</span>
                                </div>
                                <div class="chunk-body">{text_body}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if final_timings:
                thtml = "".join(
                    f'<span class="timing-badge">{k}: <b>{v*1000:.1f}ms</b></span>'
                    for k, v in final_timings.items()
                )
                st.markdown(f"<div style='margin-bottom:12px;'>{thtml}</div>", unsafe_allow_html=True)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_text,
                    "chunks": final_chunks,
                    "timings": final_timings,
                }
            )

        except Exception as ex:
            st.error(f"Execution Error: {ex}")


# ═══════════════════════════════════════════════════════════════
# TAB 2: Evaluation Benchmarks
# ═══════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("### Ragas Offline Benchmark Evaluation")
    st.markdown("Empirical comparison across 15 ground-truth evaluation queries evaluating retrieval precision, recall, and answer faithfulness.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Naive Vector Precision</div>
                <div class="metric-value">83.08%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Hybrid Precision</div>
                <div class="metric-value">94.67%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Hybrid + Rerank Precision</div>
                <div class="metric-value metric-value-accent">95.71%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Precision Delta</div>
                <div class="metric-value" style="color:#38bdf8;">+12.63%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Detailed Metric Comparison Matrix")
    benchmark_data = {
        "Retrieval Strategy": [
            "Naive (Dense Vector Search)",
            "Hybrid (Dense Vector + BM25 Fusion)",
            "Hybrid + Cross-Encoder Reranker",
        ],
        "Context Precision": ["83.08%", "94.67%", "95.71%"],
        "Context Recall": ["100.00%", "99.78%", "99.76%"],
        "Faithfulness": ["58.87%", "53.24%", "52.80%"],
        "Average Retrieval Latency": ["~70 ms", "~75 ms", "~290 ms"],
        "Pipeline Trade-off": [
            "Baseline semantic vector similarity; susceptible to lexical misses.",
            "Captures both semantic and exact keyword matches using RRF (k=60).",
            "Highest precision; Cross-Encoder scores exact query-document token interactions.",
        ],
    }
    st.table(benchmark_data)


# ═══════════════════════════════════════════════════════════════
# TAB 3: Indexed Documents
# ═══════════════════════════════════════════════════════════════
with tab_corpus:
    st.markdown("### Indexed Corpus Explorer")
    st.markdown("Files located in `data/` currently processed into chunk embeddings and inverted lexical index.")

    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in sorted(os.listdir(data_dir)) if f.endswith((".md", ".txt", ".pdf"))]
        for fname in files:
            fpath = os.path.join(data_dir, fname)
            fsize = os.path.getsize(fpath)
            with st.expander(f"{fname} ({fsize} bytes)", expanded=False):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        st.code(f.read(), language="markdown")
                except Exception as e:
                    st.error(f"Cannot read file: {e}")
    else:
        st.info("No documents found in directory.")
