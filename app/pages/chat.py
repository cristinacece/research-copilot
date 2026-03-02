# -*- coding: utf-8 -*-
"""
Chat page – multi-turn RAG conversation with strategy selector.
"""

import os
import sys
import time

import streamlit as st

# Path setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.rag_pipeline import query as rag_query
from src.generation.generator import STRATEGIES
from app.components.styles import ACADEMIC_CSS

st.markdown(ACADEMIC_CSS, unsafe_allow_html=True)
st.title("💬 Chat con los Papers")

# ── Guards ──────────────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Configura tu OpenAI API Key en la página **Configuración**.")
    st.stop()

# ── Controls ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    strategy = st.selectbox(
        "Estrategia de prompting",
        list(STRATEGIES.keys()),
        help=(
            "**Delimitadores**: Contexto enmarcado con ###.\n\n"
            "**JSON Estructurado**: Respuesta como hipótesis + variables + mecanismo.\n\n"
            "**Few-Shot**: Calibrado con ejemplos académicos.\n\n"
            "**Chain-of-Thought**: Razonamiento causal en 4 pasos."
        ),
    )
with col2:
    n_chunks = st.slider("Chunks a recuperar", 1, 6, 3)
with col3:
    chunk_config = st.selectbox("Tamaño de chunk", ["small", "large"])

st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ── Render existing history ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📚 Citas APA"):
                for cite in msg["citations"]:
                    st.markdown(f'<div class="citation-block">{cite}</div>', unsafe_allow_html=True)
        if msg.get("chunks"):
            with st.expander("🔍 Chunks recuperados"):
                for c in msg["chunks"]:
                    sim = c.get("similarity_score", 0)
                    title = c.get("paper_title", "—")
                    st.markdown(
                        f"**{title}** "
                        f'<span class="sim-badge">{sim:.2f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(c["text"][:300] + "…" if len(c["text"]) > 300 else c["text"])
        if msg.get("meta"):
            st.caption(
                f"Estrategia: {msg['meta']['strategy']} · "
                f"Latencia: {msg['meta']['latency_s']}s · "
                f"Config: {msg['meta']['chunk_config']}"
            )

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.chat_input("Escribe tu pregunta de investigación…")

if question:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build multi-turn context from last 4 exchanges
    history_context = ""
    recent = [m for m in st.session_state.messages if m["role"] == "user"][-4:]
    if len(recent) > 1:
        history_context = "\n".join(
            f"Pregunta previa: {m['content']}" for m in recent[:-1]
        ) + "\n\n"

    full_question = history_context + question if history_context else question

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner(f"Consultando corpus con estrategia *{strategy}*…"):
            try:
                result = rag_query(
                    question=full_question,
                    strategy=strategy,
                    n=n_chunks,
                    chunk_config=chunk_config,
                )
                answer    = result["answer"]
                citations = result["citations"]
                chunks    = result["chunks_used"]
                latency   = result["latency_s"]

                st.markdown(answer)

                if citations:
                    with st.expander("📚 Citas APA"):
                        for cite in citations:
                            st.markdown(
                                f'<div class="citation-block">{cite}</div>',
                                unsafe_allow_html=True,
                            )

                if chunks:
                    with st.expander("🔍 Chunks recuperados"):
                        for c in chunks:
                            sim = c.get("similarity_score", 0)
                            title = c.get("paper_title", "—")
                            st.markdown(
                                f"**{title}** "
                                f'<span class="sim-badge">{sim:.2f}</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                c["text"][:300] + "…" if len(c["text"]) > 300 else c["text"]
                            )

                st.caption(
                    f"Estrategia: {strategy} · Latencia: {latency}s · Config: {chunk_config}"
                )

                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  answer,
                    "citations": citations,
                    "chunks":   chunks,
                    "meta": {
                        "strategy":     strategy,
                        "latency_s":    latency,
                        "chunk_config": chunk_config,
                    },
                })
                st.session_state.query_count += 1

            except Exception as exc:
                st.error(f"Error al procesar la pregunta: {exc}")

# ── Footer controls ───────────────────────────────────────────────────────────
if st.session_state.messages:
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("🗑️ Limpiar historial"):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        st.caption(f"Consultas en esta sesión: {st.session_state.query_count}")
