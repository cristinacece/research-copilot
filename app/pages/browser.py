# -*- coding: utf-8 -*-
"""
Paper Browser page – search and filter the corpus.
"""

import os
import sys
import json

import streamlit as st
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from app.components.citation import format_apa
from app.components.styles import ACADEMIC_CSS

st.markdown(ACADEMIC_CSS, unsafe_allow_html=True)
st.title("📖 Catálogo de Papers")

CATALOG_PATH = os.path.join(BASE_DIR, "papers", "paper_catalog.json")


@st.cache_data
def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["papers"]


if not os.path.exists(CATALOG_PATH):
    st.error("Catálogo no encontrado. Ejecuta `python src/rag_pipeline.py` primero.")
    st.stop()

papers = load_catalog()
df = pd.DataFrame(papers)

# ── Search bar ───────────────────────────────────────────────────────────────
search = st.text_input("🔍 Buscar por título, autor o año", placeholder="e.g. soberanía alimentaria 2020")

# ── Filters ──────────────────────────────────────────────────────────────────
with st.expander("Filtros avanzados", expanded=False):
    f_col1, f_col2, f_col3 = st.columns(3)

    years = sorted(df["year"].dropna().astype(int).unique())
    with f_col1:
        year_range = st.slider(
            "Rango de años",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years))),
        )

    all_topics = sorted({t.strip() for p in papers for t in p.get("topics", []) if t.strip()})
    with f_col2:
        sel_topics = st.multiselect("Tópicos", all_topics)

    all_authors = sorted({a.strip() for p in papers for a in p.get("authors", []) if a.strip()})
    with f_col3:
        sel_authors = st.multiselect("Autores", all_authors)

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df[df["year"].astype(int).between(year_range[0], year_range[1])].copy()

if sel_topics:
    filtered = filtered[filtered["topics"].apply(
        lambda ts: any(t in ts for t in sel_topics)
    )]
if sel_authors:
    filtered = filtered[filtered["authors"].apply(
        lambda aths: any(a in aths for a in sel_authors)
    )]
if search:
    q = search.lower()
    mask = (
        filtered["title"].str.lower().str.contains(q, na=False)
        | filtered["authors"].apply(lambda a: q in str(a).lower())
        | filtered["year"].astype(str).str.contains(q)
    )
    filtered = filtered[mask]

st.markdown(f"**{len(filtered)}** de **{len(df)}** papers")
st.markdown("---")

# ── Paper cards ───────────────────────────────────────────────────────────────
for _, paper in filtered.iterrows():
    header = f"📄 **{paper['title']}** — {int(paper['year'])}"
    with st.expander(header, expanded=False):
        left, right = st.columns([3, 2])

        with left:
            autores_str = ", ".join(paper["authors"]) if paper.get("authors") else "—"
            st.markdown(f"**Autores:** {autores_str}")
            if paper.get("venue"):
                st.markdown(f"**Venue:** {paper['venue']}")
            st.markdown(f"**Páginas:** {int(paper['pages'])}")
            temas_str = ", ".join(paper["topics"]) if paper.get("topics") else "—"
            st.markdown(f"**Tópicos:** {temas_str}")
            if paper.get("doi"):
                doi_val = str(paper["doi"]).strip()
                if doi_val.startswith("http"):
                    st.markdown(f"**URL:** [{doi_val[:60]}...]({doi_val})")
                else:
                    st.markdown(f"**DOI:** {doi_val}")
            st.markdown(f"**Archivo:** `{paper['filename']}`")

            # APA citation
            apa = format_apa({
                "authors":     paper["authors"],
                "year":        paper["year"],
                "paper_title": paper["title"],
                "venue":       paper.get("venue", ""),
                "doi":         paper.get("doi", ""),
            })
            st.markdown("**Cita APA:**")
            st.markdown(f'<div class="citation-block">{apa}</div>', unsafe_allow_html=True)

        with right:
            st.markdown("**Abstract:**")
            st.markdown(
                f'<div class="abstract-block">{paper["abstract"]}</div>',
                unsafe_allow_html=True,
            )
