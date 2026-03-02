# -*- coding: utf-8 -*-
"""
Tests for src/retrieval/ module.
Uses a lightweight mock so no API calls are needed.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_chunks(n: int = 3) -> list[dict]:
    return [
        {
            "chunk_id":         f"paper_00{i}_c0000_small",
            "text":             f"Texto de ejemplo del chunk {i} sobre seguridad alimentaria.",
            "paper_id":         f"paper_00{i}",
            "paper_title":      f"Paper de Prueba {i}",
            "authors":          "Autor Uno, Autor Dos",
            "year":             2020 + i,
            "page_number":      i * 2,
            "similarity_score": round(0.9 - i * 0.05, 2),
        }
        for i in range(1, n + 1)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRetrieve:
    @patch("src.retrieval.retriever.ChromaStore")
    @patch("src.retrieval.retriever.Embedder")
    def test_retrieve_returns_tuple(self, mock_embedder_cls, mock_store_cls):
        # Setup mocks
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = _make_fake_chunks(3)
        mock_store._collection_name = "research_copilot_small"
        mock_store_cls.return_value = mock_store

        # Reset module-level globals so our mocks are used
        import src.retrieval.retriever as ret_mod
        ret_mod._embedder = None
        ret_mod._store    = None

        from src.retrieval.retriever import retrieve
        result = retrieve("¿Qué es la seguridad alimentaria?", n_results=3)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("src.retrieval.retriever.ChromaStore")
    @patch("src.retrieval.retriever.Embedder")
    def test_retrieve_returns_correct_structure(self, mock_embedder_cls, mock_store_cls):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = _make_fake_chunks(3)
        mock_store._collection_name = "research_copilot_small"
        mock_store_cls.return_value = mock_store

        import src.retrieval.retriever as ret_mod
        ret_mod._embedder = None
        ret_mod._store    = None

        from src.retrieval.retriever import retrieve
        chunks, sources = retrieve("test question", n_results=3)

        assert len(chunks) == 3
        for c in chunks:
            for key in ("chunk_id", "text", "paper_id", "paper_title", "authors", "year",
                        "page_number", "similarity_score"):
                assert key in c, f"Missing key: {key}"

    @patch("src.retrieval.retriever.ChromaStore")
    @patch("src.retrieval.retriever.Embedder")
    def test_retrieve_sources_deduplicated(self, mock_embedder_cls, mock_store_cls):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_embedder_cls.return_value = mock_embedder

        # Two chunks from the same paper
        duplicate_chunks = [
            {
                "chunk_id": "paper_001_c0000_small",
                "text": "Texto A",
                "paper_id": "paper_001",
                "paper_title": "Paper Uno",
                "authors": "Autor A",
                "year": 2022,
                "page_number": 1,
                "similarity_score": 0.9,
            },
            {
                "chunk_id": "paper_001_c0001_small",
                "text": "Texto B",
                "paper_id": "paper_001",
                "paper_title": "Paper Uno",
                "authors": "Autor A",
                "year": 2022,
                "page_number": 2,
                "similarity_score": 0.85,
            },
        ]

        mock_store = MagicMock()
        mock_store.search.return_value = duplicate_chunks
        mock_store._collection_name = "research_copilot_small"
        mock_store_cls.return_value = mock_store

        import src.retrieval.retriever as ret_mod
        ret_mod._embedder = None
        ret_mod._store    = None

        from src.retrieval.retriever import retrieve
        chunks, sources = retrieve("test", n_results=2)

        assert len(sources) == 1, "Sources should be deduplicated"
        assert sources[0] == "Paper Uno"

    @patch("src.retrieval.retriever.ChromaStore")
    @patch("src.retrieval.retriever.Embedder")
    def test_retrieve_similarity_scores_present(self, mock_embedder_cls, mock_store_cls):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.0] * 1536
        mock_embedder_cls.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = _make_fake_chunks(3)
        mock_store._collection_name = "research_copilot_small"
        mock_store_cls.return_value = mock_store

        import src.retrieval.retriever as ret_mod
        ret_mod._embedder = None
        ret_mod._store    = None

        from src.retrieval.retriever import retrieve
        chunks, _ = retrieve("question", n_results=3)

        for c in chunks:
            assert isinstance(c["similarity_score"], float)
            assert 0.0 <= c["similarity_score"] <= 1.0
