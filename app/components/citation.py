# -*- coding: utf-8 -*-
"""
APA 7 citation formatter — returns HTML-safe string.
Title is wrapped in <em>, DOI becomes a clickable <a>.
"""

import html as _html


def format_apa(chunk_meta: dict) -> str:
    """
    Format chunk metadata as an APA 7 reference string.

    Expected keys in chunk_meta:
        authors      – str or list of author names
        year         – int or str
        paper_title  – str (title of the paper)
        venue        – str (journal/conference/source, optional)
        doi          – str (optional)
    """
    # --- Authors ---
    authors_raw = chunk_meta.get("authors", "")
    year        = chunk_meta.get("year", "s.f.")
    title       = _html.escape(str(chunk_meta.get("paper_title", chunk_meta.get("title", "Sin título"))))
    venue       = _html.escape(str(chunk_meta.get("venue", "")))
    doi         = chunk_meta.get("doi", "")

    if isinstance(authors_raw, list):
        parts = [a.strip() for a in authors_raw if a.strip()]
    else:
        parts = [a.strip() for a in str(authors_raw).split(",") if a.strip()]

    if not parts:
        author_str = "Autor desconocido"
    elif len(parts) == 1:
        author_str = _html.escape(parts[0])
    elif len(parts) == 2:
        author_str = f"{_html.escape(parts[0])} y {_html.escape(parts[1])}"
    elif len(parts) <= 20:
        author_str = ", ".join(_html.escape(p) for p in parts[:-1]) + f" y {_html.escape(parts[-1])}"
    else:
        author_str = ", ".join(_html.escape(p) for p in parts[:19]) + f", ... y {_html.escape(parts[-1])}"

    # --- Build APA string (HTML-safe: <em> for italics, <a> for DOI) ---
    apa = f"{author_str} ({year}). <em>{title}</em>."
    if venue:
        apa += f" {venue}."
    if doi:
        doi_val = doi.strip()
        if not doi_val.startswith("http"):
            doi_val = f"https://doi.org/{doi_val}"
        apa += f' <a href="{doi_val}" target="_blank">{doi_val}</a>'

    return apa
