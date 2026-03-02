# -*- coding: utf-8 -*-
"""
PDF extraction using PyMuPDF (fitz).
"""

import os
from typing import Optional
import fitz  # PyMuPDF


def extract_pdf(path: str) -> dict:
    """
    Extract text and metadata from a PDF file.

    Returns:
        dict with keys:
            text           – full extracted text (str)
            metadata       – dict with title, author, subject, creator, etc.
            pages          – list of per-page text strings
            total_pages    – number of pages (int)
            extraction_warnings – list of warning strings
    """
    warnings: list[str] = []
    pages_text: list[str] = []
    metadata: dict = {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        doc = fitz.open(path)
        metadata = {
            "title":    doc.metadata.get("title", ""),
            "author":   doc.metadata.get("author", ""),
            "subject":  doc.metadata.get("subject", ""),
            "creator":  doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
        }

        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages_text.append(text)
            if len(text.strip()) < 20:
                warnings.append(f"Page {page_num + 1} has very little text (possibly scanned).")

        doc.close()

    except Exception as exc:
        warnings.append(f"Extraction error: {exc}")
        return {
            "text": "",
            "metadata": metadata,
            "pages": pages_text,
            "total_pages": 0,
            "extraction_warnings": warnings,
        }

    full_text = "\n".join(pages_text)
    if len(full_text.strip()) < 100:
        warnings.append("Extracted text is very short – PDF may be image-based or encrypted.")

    return {
        "text": full_text,
        "metadata": metadata,
        "pages": pages_text,
        "total_pages": len(pages_text),
        "extraction_warnings": warnings,
    }
