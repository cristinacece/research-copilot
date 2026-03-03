# -*- coding: utf-8 -*-
"""
APA 7 citation formatter — returns a Markdown string (no raw HTML).

APA 7 format for reports / working papers:
    Last, F. M., & Last, F. M. (Year). *Title of report*. Publisher/Series. URL

Italics: title only. Venue (publisher/series) in plain text.
"""

import re

# Words that signal an organizational name rather than a person's name
# e.g. "FAO e IFPRI", "FAO y UNICEF", "FAO & WFP"
_ORG_CONNECTORS = {"e", "y", "&", "and", "et", "und", "ou"}


def _is_org_name(name: str) -> bool:
    """Return True if name looks like an institutional/organizational author."""
    tokens = name.strip().split()
    # Single all-caps token → organization (e.g. "FAO", "IFPRI", "OMS")
    if len(tokens) == 1:
        return tokens[0].isupper() and len(tokens[0]) > 1
    # Multi-token: if any token is a known connector, it's a compound org name
    # e.g. "FAO e IFPRI", "FAO & World Food Programme"
    return any(t.lower() in _ORG_CONNECTORS for t in tokens)


def _to_apa_name(name: str) -> str:
    """Convert 'First [Middle] Last' → 'Last, F.' per APA 7.

    Examples:
        'Mariana Escobar Arango' → 'Escobar Arango, M.'
        'T.D. Brewer'            → 'Brewer, T.D.'
        'FAO'                    → 'FAO'         (org kept as-is)
        'FAO e IFPRI'            → 'FAO e IFPRI' (org kept as-is)
        'Lester R. Brown'        → 'Brown, L. R.'
    """
    name = name.strip()
    tokens = name.split()

    # Single token or organizational name → return as-is
    if len(tokens) == 1 or _is_org_name(name):
        return name

    # Person's name: last name is last token; collect initials from remaining tokens
    last = tokens[-1]
    given = tokens[:-1]  # first + optional middle names

    initials = []
    for part in given:
        # Already an initial like 'T.D.' or 'R.' → keep as-is
        if len(part) <= 5 and "." in part:
            initials.append(part)
        else:
            initials.append(part[0].upper() + ".")

    return f"{last}, {' '.join(initials)}"


def _parse_author_string(raw: str) -> list[str]:
    """Parse a comma-separated author string into individual author names.

    Handles:
    - 'First Last, First Last'       → ['First Last', 'First Last']
    - 'Last, F., Last, F.'           → ['Last, F.', 'Last, F.']
    - 'Akbari, M., Foroudi, P.'      → ['Akbari, M.', 'Foroudi, P.']
    - 'FAO, FIDA, OMS'               → ['FAO', 'FIDA', 'OMS']
    - 'FAO e IFPRI'                  → ['FAO e IFPRI']
    """
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    result = []
    i = 0
    # Matches bare initial tokens like 'M.', 'A.J.', 'M.T.'
    _initial_re = re.compile(r'^[A-Z][A-Za-z]?\.([A-Z]\.)*$')
    while i < len(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if nxt and _initial_re.match(nxt):
            # e.g. 'Akbari' + 'M.' → 'Akbari, M.'
            result.append(f"{tokens[i]}, {nxt}")
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def _format_single_author(name: str) -> str:
    """Return APA 7 name. Names already in 'Last, F.' format used as-is."""
    if "," in name and not _is_org_name(name):
        return name  # already formatted
    return _to_apa_name(name)


def format_apa(chunk_meta: dict) -> str:
    """
    Format chunk metadata as an APA 7 reference in Markdown syntax.

    APA 7 for reports/working papers:
        Last, F. (Year). *Title*. Publisher/Series. URL

    Only the title is italicised. Venue is plain text.
    """
    authors_raw = chunk_meta.get("authors", "")
    year        = chunk_meta.get("year", "s.f.")
    title       = str(chunk_meta.get("paper_title", chunk_meta.get("title", "Sin título"))).strip()
    venue       = str(chunk_meta.get("venue", "")).strip()
    doi         = str(chunk_meta.get("doi", "")).strip()

    # ── Parse & format authors ─────────────────────────────────────────────
    if isinstance(authors_raw, list):
        raw_list = [a.strip() for a in authors_raw if a.strip()]
    else:
        raw_list = _parse_author_string(str(authors_raw))

    parts = [_format_single_author(a) for a in raw_list]

    if not parts:
        author_str = "Autor desconocido"
    elif len(parts) == 1:
        author_str = parts[0]
    elif len(parts) == 2:
        author_str = f"{parts[0]}, & {parts[1]}"
    elif len(parts) <= 20:
        author_str = ", ".join(parts[:-1]) + f", & {parts[-1]}"
    else:
        # APA 7: first 19, ellipsis, last author
        author_str = ", ".join(parts[:19]) + f", ... & {parts[-1]}"

    # ── Assemble Markdown reference ────────────────────────────────────────
    # Title in italics (APA 7 for reports / working papers)
    # Venue in plain text (publisher/series — NOT italicised)
    apa = f"{author_str} ({year}). *{title}*."

    if venue:
        apa += f" {venue}."

    # DOI / URL as plain clickable link
    if doi:
        url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        apa += f" {url}"

    return apa
