# -*- coding: utf-8 -*-
"""
Analytics page – corpus statistics and usage metrics.
"""

import os
import sys
import json
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from app.components.styles import ACADEMIC_CSS

st.markdown(ACADEMIC_CSS, unsafe_allow_html=True)
st.title("📊 Analytics del Corpus")

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
df["year"] = df["year"].astype(int)

# ── Session usage stats ────────────────────────────────────────────────────
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

st.subheader("Estadísticas de uso (sesión actual)")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Papers indexados", len(df))
m_col2.metric("Consultas realizadas", st.session_state.query_count)
m_col3.metric("Rango temporal", f"{df['year'].min()} – {df['year'].max()}")

st.markdown("---")

# ── Row 1: Papers por año + Tópicos ──────────────────────────────────────────
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Papers por año")
    year_counts = df["year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["Año", "Cantidad"]
    fig_years = px.bar(
        year_counts,
        x="Año",
        y="Cantidad",
        color="Cantidad",
        color_continuous_scale="Blues",
        title="Distribución temporal del corpus",
    )
    fig_years.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_years, use_container_width=True)

with row1_col2:
    st.subheader("Tópicos más frecuentes")
    all_topics = [t.strip() for p in papers for t in p.get("topics", []) if t.strip()]
    topic_counts = Counter(all_topics).most_common(15)
    topics_df = pd.DataFrame(topic_counts, columns=["Tópico", "Frecuencia"])
    fig_topics = px.bar(
        topics_df,
        x="Frecuencia",
        y="Tópico",
        orientation="h",
        color="Frecuencia",
        color_continuous_scale="Greens",
        title="Top 15 tópicos más frecuentes",
    )
    fig_topics.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_topics, use_container_width=True)

# ── Row 2: Timeline scatter ───────────────────────────────────────────────────
st.subheader("Línea de tiempo del corpus")
timeline_df = df.copy()
timeline_df["autores_str"] = timeline_df["authors"].apply(
    lambda a: (", ".join(a[:2]) + " et al.") if isinstance(a, list) and len(a) > 2
    else (", ".join(a) if isinstance(a, list) else str(a))
)
timeline_df = timeline_df.sort_values("year").reset_index(drop=True)
timeline_df["idx"] = timeline_df.index

fig_timeline = px.scatter(
    timeline_df,
    x="year",
    y="idx",
    hover_name="title",
    hover_data={"autores_str": True, "year": True, "idx": False},
    title="Publicaciones ordenadas por año",
    labels={"year": "Año de publicación", "idx": "Paper"},
    color="year",
    color_continuous_scale="Viridis",
)
fig_timeline.update_traces(marker=dict(size=14, opacity=0.8))
fig_timeline.update_layout(
    coloraxis_showscale=False,
    yaxis=dict(showticklabels=False, title=""),
)
st.plotly_chart(fig_timeline, use_container_width=True)

# ── Row 3: Summary table ──────────────────────────────────────────────────────
st.subheader("Tabla resumen del corpus")
table_df = df[["title", "authors", "year", "venue", "topics"]].copy()
table_df["authors"] = table_df["authors"].apply(
    lambda a: (", ".join(a[:2]) + (" et al." if len(a) > 2 else ""))
    if isinstance(a, list) else str(a)
)
table_df["topics"] = table_df["topics"].apply(
    lambda t: ", ".join(t[:3]) if isinstance(t, list) else str(t)
)
table_df = table_df.rename(columns={
    "title":   "Título",
    "authors": "Autores",
    "year":    "Año",
    "venue":   "Venue",
    "topics":  "Tópicos",
})
st.dataframe(table_df, use_container_width=True, hide_index=True)
