# -*- coding: utf-8 -*-
"""
Research Copilot – Streamlit entry point.

Run with:
    streamlit run app/main.py
"""

import os
import sys

# Ensure project root is in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global session state defaults ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "chunk_config" not in st.session_state:
    st.session_state.chunk_config = "small"

# ── Navigation ────────────────────────────────────────────────────────────────
pages = {
    "💬 Chat":          "app/pages/chat.py",
    "📖 Paper Browser": "app/pages/browser.py",
    "📊 Analytics":     "app/pages/analytics.py",
    "⚙️ Configuración": "app/pages/settings.py",
}

with st.sidebar:
    st.title("📚 Research Copilot")
    st.markdown(
        "Sistema RAG para papers sobre **seguridad alimentaria** "
        "y comercio internacional."
    )
    st.markdown("---")

    selection = st.radio(
        "Navegación",
        list(pages.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick status
    chroma_path = os.path.join(BASE_DIR, "chroma_db")
    catalog_path = os.path.join(BASE_DIR, "papers", "paper_catalog.json")

    if os.path.exists(chroma_path):
        st.success("✅ ChromaDB lista")
    else:
        st.warning("⚠️ ChromaDB no encontrada")

    if os.path.exists(catalog_path):
        import json
        with open(catalog_path, encoding="utf-8") as f:
            count = len(json.load(f)["papers"])
        st.info(f"📄 {count} papers indexados")
    else:
        st.error("❌ Catálogo no encontrado")

    if os.environ.get("OPENAI_API_KEY"):
        st.success("🔑 API Key configurada")
    else:
        st.warning("🔑 API Key no configurada")

    st.markdown("---")
    st.caption("Cristina Celeste Tamay Blanco · 2025")

# ── Load selected page ────────────────────────────────────────────────────────
page_file = pages[selection]
page_path = os.path.join(BASE_DIR, page_file)

with open(page_path, "r", encoding="utf-8") as f:
    page_code = f.read()

exec(compile(page_code, page_path, "exec"), {"__name__": "__main__"})
