# -*- coding: utf-8 -*-
"""
APA 7 citation formatter.
"""


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
    title       = chunk_meta.get("paper_title", chunk_meta.get("title", "Sin título"))
    venue       = chunk_meta.get("venue", "")
    doi         = chunk_meta.get("doi", "")

    if isinstance(authors_raw, list):
        parts = [a.strip() for a in authors_raw if a.strip()]
    else:
        parts = [a.strip() for a in str(authors_raw).split(",") if a.strip()]

    if not parts:
        author_str = "Autor desconocido"
    elif len(parts) == 1:
        author_str = parts[0]
    elif len(parts) == 2:
        author_str = f"{parts[0]} y {parts[1]}"
    elif len(parts) <= 20:
        author_str = ", ".join(parts[:-1]) + f" y {parts[-1]}"
    else:
        author_str = ", ".join(parts[:19]) + ", ... y " + parts[-1]

    # --- Build APA string ---
    apa = f"{author_str} ({year}). *{title}*."
    if venue:
        apa += f" {venue}."
    if doi:
        doi_val = doi.strip()
        if doi_val.startswith("http"):
            apa += f" {doi_val}"
        else:
            apa += f" https://doi.org/{doi_val}"

    return apa
