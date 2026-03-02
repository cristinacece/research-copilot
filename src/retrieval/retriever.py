# -*- coding: utf-8 -*-
"""
Retrieval layer: embeds a question and queries ChromaDB.
"""

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embedding.embedder import Embedder
from src.vectorstore.chroma_store import ChromaStore

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

_embedder: Embedder       = None
_store:    ChromaStore     = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_store(chunk_config: str = "small") -> ChromaStore:
    global _store
    collection_name = f"research_copilot_{chunk_config}"
    if _store is None or _store._collection_name != collection_name:
        _store = ChromaStore(path=CHROMA_PATH, collection_name=collection_name)
    return _store


def retrieve(
    question: str,
    n_results: int = 3,
    chunk_config: str = "small",
) -> tuple[list[dict], list[str]]:
    """
    Embed the question and retrieve the top-n matching chunks.

    Returns:
        chunks  – list of dicts: chunk_id, text, paper_id, paper_title,
                  authors, year, page_number, similarity_score
        sources – deduplicated list of paper_title strings
    """
    embedder = _get_embedder()
    store    = _get_store(chunk_config)

    query_vec = embedder.embed_query(question)
    hits      = store.search(query_vec, n_results=n_results)

    # Build clean output list
    chunks: list[dict] = []
    for h in hits:
        chunks.append({
            "chunk_id":         h.get("chunk_id", ""),
            "text":             h.get("text", ""),
            "paper_id":         h.get("paper_id", ""),
            "paper_title":      h.get("paper_title", ""),
            "authors":          h.get("authors", ""),
            "year":             h.get("year", 0),
            "page_number":      h.get("page_number", 0),
            "venue":            h.get("venue", ""),
            "doi":              h.get("doi", ""),
            "similarity_score": h.get("similarity_score", 0.0),
        })

    sources = list(dict.fromkeys(c["paper_title"] for c in chunks if c["paper_title"]))
    return chunks, sources
