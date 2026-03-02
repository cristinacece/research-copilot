# -*- coding: utf-8 -*-
"""
Settings page – API key, chunk config, system info.
"""

import os
import sys

import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from app.components.styles import ACADEMIC_CSS

st.markdown(ACADEMIC_CSS, unsafe_allow_html=True)
st.title("⚙️ Configuración")

# ── API Key ───────────────────────────────────────────────────────────────────
st.subheader("🔑 API Key de OpenAI")
api_key_input = st.text_input(
    "OpenAI API Key",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", ""),
    help="Tu clave de API de OpenAI. Se guarda solo en memoria (no se escribe al disco).",
)
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input
    st.success("API Key guardada en memoria para esta sesión.")

st.info(
    "Para persistir la API Key entre sesiones, crea un archivo `.env` "
    "en la raíz del proyecto con el contenido:\n\n"
    "```\nOPENAI_API_KEY=sk-...tu-clave-aquí\n```"
)

st.markdown("---")

# ── Chunk config ──────────────────────────────────────────────────────────────
st.subheader("🧩 Configuración de Chunks")
if "chunk_config" not in st.session_state:
    st.session_state.chunk_config = "small"

st.session_state.chunk_config = st.radio(
    "Tamaño de chunk activo",
    ["small", "large"],
    index=0 if st.session_state.chunk_config == "small" else 1,
    help="**small** (256 tokens, 50 overlap) — mayor precisión.\n\n**large** (1024 tokens, 100 overlap) — más contexto por chunk.",
)

st.markdown("---")

# ── System info ───────────────────────────────────────────────────────────────
st.subheader("🖥️ Info del sistema")

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
chroma_exists = os.path.exists(CHROMA_PATH)

info_col1, info_col2 = st.columns(2)
with info_col1:
    if chroma_exists:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collections = client.list_collections()
            st.success(f"ChromaDB: {len(collections)} colección(es)")
            for col in collections:
                coll = client.get_collection(col.name)
                st.write(f"  • **{col.name}**: {coll.count()} chunks")
        except Exception as e:
            st.warning(f"ChromaDB encontrada pero no accesible: {e}")
    else:
        st.error("ChromaDB no encontrada. Ejecuta el pipeline primero.")

with info_col2:
    try:
        import chromadb as cdb
        st.write(f"**ChromaDB version:** {cdb.__version__}")
    except Exception:
        pass
    try:
        import openai as oai
        st.write(f"**OpenAI SDK version:** {oai.__version__}")
    except Exception:
        pass
    try:
        import streamlit as _st
        st.write(f"**Streamlit version:** {_st.__version__}")
    except Exception:
        pass

st.markdown("---")
st.caption("Research Copilot · Cristina Celeste Tamay Blanco · 2025")
