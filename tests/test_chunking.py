# -*- coding: utf-8 -*-
"""
Tests for src/chunking/ module.
"""

import os
import sys
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.chunking.chunker import chunk_text, CHUNK_CONFIGS
import tiktoken


SAMPLE_TEXT = (
    "La seguridad alimentaria es un concepto fundamental en las relaciones internacionales. "
    "Se refiere a la disponibilidad, acceso, utilización y estabilidad de los alimentos. "
    "El comercio internacional juega un papel crucial en la distribución global de alimentos. "
    "Los países en desarrollo son especialmente vulnerables a las fluctuaciones de precios. "
    "El cambio climático agrava las amenazas existentes a los sistemas agroalimentarios. "
    "Las políticas de subsidios distorsionan los mercados internacionales de commodities. "
    "La soberanía alimentaria propone el derecho de los pueblos a definir sus propias políticas. "
    "Las organizaciones multilaterales coordinan esfuerzos para reducir el hambre mundial. "
    "La pandemia de COVID-19 reveló fragilidades estructurales en las cadenas de suministro. "
    "El conflicto en Ucrania generó disrupciones en la oferta global de granos y fertilizantes. "
) * 20  # Repeat to get enough tokens


SAMPLE_META = {
    "paper_id":    "test_001",
    "paper_title": "Test Paper",
    "authors":     "Test Author",
    "year":        2024,
}


class TestChunker:
    def test_chunk_configs_exist(self):
        assert "small" in CHUNK_CONFIGS
        assert "large" in CHUNK_CONFIGS
        assert CHUNK_CONFIGS["small"]["size"]    == 256
        assert CHUNK_CONFIGS["small"]["overlap"] == 50
        assert CHUNK_CONFIGS["large"]["size"]    == 1024
        assert CHUNK_CONFIGS["large"]["overlap"] == 100

    def test_chunk_returns_list(self):
        result = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="small")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_has_required_keys(self):
        chunks = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="small")
        for key in ("chunk_id", "text", "token_count", "start_token", "end_token", "metadata"):
            assert key in chunks[0], f"Missing key: {key}"

    def test_small_token_count_correct(self):
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        chunks = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="small")
        for c in chunks:
            actual_tokens = len(enc.encode(c["text"]))
            # Allow ±1 due to decode rounding
            assert abs(actual_tokens - c["token_count"]) <= 1, (
                f"Token count mismatch: reported {c['token_count']}, actual {actual_tokens}"
            )

    def test_large_token_count_correct(self):
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        chunks = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="large")
        for c in chunks:
            actual_tokens = len(enc.encode(c["text"]))
            assert abs(actual_tokens - c["token_count"]) <= 1

    def test_small_max_tokens(self):
        chunks = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="small")
        for c in chunks:
            assert c["token_count"] <= 256, f"Chunk exceeds max tokens: {c['token_count']}"

    def test_large_max_tokens(self):
        chunks = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="large")
        for c in chunks:
            assert c["token_count"] <= 1024, f"Chunk exceeds max tokens: {c['token_count']}"

    def test_chunk_id_includes_config(self):
        chunks_s = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="small")
        chunks_l = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="large")
        assert "small" in chunks_s[0]["chunk_id"]
        assert "large" in chunks_l[0]["chunk_id"]

    def test_metadata_preserved(self):
        chunks = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="small")
        for c in chunks:
            assert c["metadata"]["paper_id"]    == "test_001"
            assert c["metadata"]["chunk_config"] == "small"

    def test_empty_text_returns_empty_list(self):
        result = chunk_text("", SAMPLE_META, config="small")
        assert result == []

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            chunk_text(SAMPLE_TEXT, SAMPLE_META, config="nonexistent")

    def test_large_produces_fewer_chunks(self):
        chunks_s = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="small")
        chunks_l = chunk_text(SAMPLE_TEXT, SAMPLE_META, config="large")
        assert len(chunks_l) < len(chunks_s), "Large config should produce fewer chunks"
