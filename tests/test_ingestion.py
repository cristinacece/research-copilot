# -*- coding: utf-8 -*-
"""
Tests for src/ingestion/ module.
"""

import os
import sys
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.ingestion.pdf_extractor import extract_pdf
from src.ingestion.text_cleaner import clean_text

PAPERS_DIR = os.path.join(BASE_DIR, "papers")


def _first_pdf() -> str:
    """Return the path of the first PDF found in papers/."""
    for fn in sorted(os.listdir(PAPERS_DIR)):
        if fn.endswith(".pdf"):
            return os.path.join(PAPERS_DIR, fn)
    return None


class TestPdfExtractor:
    def test_extract_returns_dict(self):
        pdf_path = _first_pdf()
        if pdf_path is None:
            pytest.skip("No PDF files found in papers/")
        result = extract_pdf(pdf_path)
        assert isinstance(result, dict)

    def test_extract_has_required_keys(self):
        pdf_path = _first_pdf()
        if pdf_path is None:
            pytest.skip("No PDF files found in papers/")
        result = extract_pdf(pdf_path)
        for key in ("text", "metadata", "pages", "total_pages", "extraction_warnings"):
            assert key in result, f"Missing key: {key}"

    def test_extract_text_is_nonempty(self):
        pdf_path = _first_pdf()
        if pdf_path is None:
            pytest.skip("No PDF files found in papers/")
        result = extract_pdf(pdf_path)
        assert len(result["text"].strip()) > 50, "Extracted text is too short"

    def test_extract_pages_count(self):
        pdf_path = _first_pdf()
        if pdf_path is None:
            pytest.skip("No PDF files found in papers/")
        result = extract_pdf(pdf_path)
        assert result["total_pages"] == len(result["pages"])
        assert result["total_pages"] > 0

    def test_extract_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_pdf("/nonexistent/path/file.pdf")


class TestTextCleaner:
    def test_clean_removes_hyphens(self):
        text = "seg-\nuridad"
        assert "seguridad" in clean_text(text)

    def test_clean_removes_isolated_page_numbers(self):
        text = "Párrafo uno.\n\n12\n\nPárrafo dos."
        cleaned = clean_text(text)
        # Isolated "12" should be removed
        import re
        assert not re.search(r"^\s*12\s*$", cleaned, re.MULTILINE)

    def test_clean_normalizes_quotes(self):
        text = "\u201cHello\u201d and \u2018world\u2019"
        cleaned = clean_text(text)
        assert '"Hello"' in cleaned
        assert "'world'" in cleaned

    def test_clean_collapses_whitespace(self):
        text = "word1   word2\t\tword3"
        cleaned = clean_text(text)
        assert "  " not in cleaned

    def test_clean_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""
