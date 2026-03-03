# -*- coding: utf-8 -*-
"""
Research Copilot – Streamlit App
Interfaz para el sistema RAG de papers sobre seguridad alimentaria.

Ejecutar:
    streamlit run app.py
"""

import os
import sys
import json
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is in path for src.* imports
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from tarea import (
    ARCHIVO_JSON,
    CHROMA_PATH,
    ESTRATEGIAS,
    get_collection,
    recuperar_contexto,
)
from src.rag_pipeline import query as rag_query
from app.components.citation import format_apa

# Map tarea.py strategy names → src pipeline strategy names
_STRATEGY_MAP = {
    "Delimitadores":     "Delimitadores",
    "JSON Estructurado": "JSON Estructurado",
    "Few-Shot Learning": "Few-Shot",
    "Chain-of-Thought":  "Chain-of-Thought",
}

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Research Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# FUNCIONES DE CARGA
# ─────────────────────────────────────────────


@st.cache_data
def load_catalog():
    with open(ARCHIVO_JSON, "r", encoding="utf-8") as f:
        return json.load(f)["papers"]


def _render_citations(fuentes: list):
    """Muestra la bibliografía en formato APA 7 al final de cada respuesta."""
    if not fuentes:
        return
    st.markdown("---")
    st.markdown("#### 📚 Referencias bibliográficas (APA 7)")
    for fuente in fuentes:
        st.markdown(fuente)
        st.markdown("")


def _render_response(content: str, fuentes: list, estrategia: str):
    """Renderiza la respuesta del asistente según la estrategia usada."""
    if estrategia == "JSON Estructurado":
        try:
            data = json.loads(content)

            # Respuesta principal
            st.markdown(data.get("answer", content))

            # Hipótesis
            if data.get("hipotesis"):
                st.markdown(f"**Hipótesis central:** {data['hipotesis']}")

            # Mecanismo de transmisión
            if data.get("mecanismo_transmision"):
                st.markdown(f"**Mecanismo de transmisión:** {data['mecanismo_transmision']}")

            # Implicaciones de políticas
            if data.get("implicaciones_politicas"):
                st.info(f"**Implicaciones para políticas:** {data['implicaciones_politicas']}")

            # Limitaciones
            if data.get("limitaciones"):
                st.markdown(f"*Limitaciones:* {data['limitaciones']}")

            # Confianza
            confidence = str(data.get("confidence", "")).lower()
            conf_map = {"high": ("success", "Alta"), "medium": ("warning", "Media"), "low": ("error", "Baja")}
            if confidence in conf_map:
                fn, label = conf_map[confidence]
                getattr(st, fn)(f"Confianza del modelo: **{label}**")

            # Variables clave
            variables = data.get("variables_clave", [])
            if variables:
                st.markdown("**Variables clave:** " + " · ".join(f"`{v}`" for v in variables))

            # Temas relacionados
            topics = data.get("related_topics", [])
            if topics:
                st.markdown("**Temas relacionados:** " + " · ".join(f"`{t}`" for t in topics))

        except (json.JSONDecodeError, TypeError):
            st.markdown(content)
    else:
        st.markdown(content)

    # Referencias APA del catálogo (todas las estrategias)
    if fuentes:
        _render_citations(fuentes)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Research Copilot")
    st.markdown("Sistema RAG para papers de seguridad alimentaria y comercio internacional.")
    st.markdown("---")

    st.subheader("⚙️ Configuración")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Tu clave de API de OpenAI. Se puede definir como variable de entorno OPENAI_API_KEY.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.markdown("---")

    # Estado de la DB
    chroma_exists = os.path.exists(CHROMA_PATH)
    if chroma_exists:
        st.success("✅ ChromaDB lista")
    else:
        st.warning("⚠️ ChromaDB no encontrada. Ejecuta `python tarea.py` primero.")

    catalog_exists = os.path.exists(ARCHIVO_JSON)
    if catalog_exists:
        papers_count = len(load_catalog())
        st.info(f"📄 {papers_count} papers indexados")
    else:
        st.error("❌ Catálogo no encontrado.")

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────

tab_chat, tab_browser, tab_viz = st.tabs(["💬 Chat", "📖 Paper Browser", "📊 Visualizaciones"])


# ══════════════════════════════════════════════
# TAB 1 – CHAT RAG
# ══════════════════════════════════════════════

with tab_chat:
    st.header("Chat con los Papers")

    # Verificaciones previas
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("Ingresa tu OpenAI API Key en el panel lateral para usar el chat.")
        st.stop()

    if not chroma_exists:
        st.error("La base de datos vectorial no existe. Ejecuta `python tarea.py` para construirla.")
        st.stop()

    # Controles
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        estrategia_nombre = st.selectbox(
            "Estrategia de prompting",
            list(ESTRATEGIAS.keys()),
            help=(
                "**Delimitadores**: Instrucciones claras con contexto delimitado por ###. "
                "Cita fuentes en formato APA.\n\n"
                "**JSON Estructurado**: Respuesta estructurada con respuesta, confianza, "
                "citas y temas relacionados.\n\n"
                "**Few-Shot Learning**: Calibrado con ejemplos de respuestas académicas "
                "con citas en texto.\n\n"
                "**Chain-of-Thought**: Razonamiento paso a paso con conclusión final "
                "y citas APA."
            ),
        )
    with ctrl_col2:
        n_resultados = st.slider("Chunks a recuperar", min_value=1, max_value=6, value=3)

    st.markdown("---")

    # Historial de conversación
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_response(
                    msg["content"],
                    msg.get("fuentes", []),
                    msg.get("estrategia", ""),
                )
                if msg.get("estrategia"):
                    st.caption(f"Estrategia: {msg['estrategia']}")
            else:
                st.markdown(msg["content"])

    # Input de usuario
    pregunta = st.chat_input("Escribe tu pregunta de investigación...")

    if pregunta:
        # Agregar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner(f"Consultando corpus con estrategia *{estrategia_nombre}*..."):
                try:
                    src_strategy = _STRATEGY_MAP.get(estrategia_nombre, "Delimitadores")
                    result = rag_query(
                        question=pregunta,
                        strategy=src_strategy,
                        n=n_resultados,
                        chunk_config="small",
                    )
                    respuesta = result["answer"]
                    chunks    = result["chunks_used"]

                    # Build APA 7 citations from retrieved chunks
                    seen: set = set()
                    fuentes: list = []
                    for c in chunks:
                        key = c.get("paper_id") or c.get("paper_title") or ""
                        if key and key not in seen:
                            seen.add(key)
                            fuentes.append(format_apa({
                                "authors":     c.get("authors", ""),
                                "year":        c.get("year", "s.f."),
                                "paper_title": c.get("paper_title", ""),
                                "venue":       c.get("venue", ""),
                                "doi":         c.get("doi", ""),
                            }))

                    _render_response(respuesta, fuentes, estrategia_nombre)
                    st.caption(
                        f"Estrategia: {estrategia_nombre} · "
                        f"Chunks: {n_resultados} · "
                        f"Latencia: {result['latency_s']}s"
                    )

                    st.session_state.messages.append({
                        "role":       "assistant",
                        "content":    respuesta,
                        "fuentes":    fuentes,
                        "estrategia": estrategia_nombre,
                    })

                except Exception as e:
                    st.error(f"Error al procesar la pregunta: {e}")

    # Botón para limpiar historial
    if st.session_state.messages:
        if st.button("🗑️ Limpiar historial", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════
# TAB 2 – PAPER BROWSER
# ══════════════════════════════════════════════

with tab_browser:
    st.header("Catálogo de Papers")

    if not catalog_exists:
        st.error("Catálogo no encontrado. Ejecuta `python tarea.py`.")
    else:
        papers = load_catalog()
        df = pd.DataFrame(papers)

        # ── Filtros ──
        st.subheader("Filtros")
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

        # ── Aplicar filtros ──
        filtered = df[df["year"].astype(int).between(year_range[0], year_range[1])]

        if sel_topics:
            filtered = filtered[filtered["topics"].apply(
                lambda ts: any(t in ts for t in sel_topics)
            )]
        if sel_authors:
            filtered = filtered[filtered["authors"].apply(
                lambda aths: any(a in aths for a in sel_authors)
            )]

        st.markdown(f"**{len(filtered)}** de **{len(df)}** papers")
        st.markdown("---")

        # ── Tarjetas de papers ──
        for _, paper in filtered.iterrows():
            header = f"📄 **{paper['title']}** — {int(paper['year'])}"
            with st.expander(header, expanded=False):
                left, right = st.columns([3, 2])

                with left:
                    autores_str = ", ".join(paper["authors"]) if paper["authors"] else "—"
                    st.markdown(f"**Autores:** {autores_str}")
                    if paper.get("venue"):
                        st.markdown(f"**Venue:** {paper['venue']}")
                    st.markdown(f"**Páginas:** {int(paper['pages'])}")
                    temas_str = ", ".join(paper["topics"]) if paper["topics"] else "—"
                    st.markdown(f"**Tópicos:** {temas_str}")
                    if paper.get("doi"):
                        doi_val = paper["doi"].strip()
                        if doi_val.startswith("http"):
                            st.markdown(f"**URL:** [{doi_val}]({doi_val})")
                        else:
                            st.markdown(f"**DOI:** {doi_val}")
                    st.markdown(f"**Archivo:** `{paper['filename']}`")

                with right:
                    st.markdown("**Abstract:**")
                    st.markdown(paper["abstract"])


# ══════════════════════════════════════════════
# TAB 3 – VISUALIZACIONES
# ══════════════════════════════════════════════

with tab_viz:
    st.header("Análisis del Corpus")

    if not catalog_exists:
        st.error("Catálogo no encontrado. Ejecuta `python tarea.py`.")
    else:
        papers = load_catalog()
        df = pd.DataFrame(papers)
        df["year"] = df["year"].astype(int)

        # ── Fila 1: Papers por año + Tópicos ──
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.subheader("Papers por año")
            year_counts = (
                df["year"].value_counts()
                .sort_index()
                .reset_index()
                .rename(columns={"index": "Año", "year": "Cantidad"})
            )
            # pandas ≥2.0 value_counts().reset_index() usa el nombre original
            if "year" in year_counts.columns and "count" in year_counts.columns:
                year_counts = year_counts.rename(columns={"year": "Año", "count": "Cantidad"})
            elif year_counts.columns.tolist() == ["index", "year"]:
                year_counts = year_counts.rename(columns={"index": "Año", "year": "Cantidad"})

            fig_years = px.bar(
                year_counts,
                x=year_counts.columns[0],
                y=year_counts.columns[1],
                color=year_counts.columns[1],
                color_continuous_scale="Blues",
                title="Distribución temporal del corpus",
                labels={year_counts.columns[0]: "Año", year_counts.columns[1]: "Cantidad"},
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

        # ── Fila 2: Línea de tiempo ──
        st.subheader("Línea de tiempo del corpus")
        timeline_df = df.copy()
        timeline_df["autores_str"] = timeline_df["authors"].apply(
            lambda a: (", ".join(a[:2]) + " et al.") if len(a) > 2 else ", ".join(a)
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

        # ── Fila 3: Tabla resumen ──
        st.subheader("Tabla de papers")
        table_df = df[["title", "authors", "year", "venue", "topics"]].copy()
        table_df["authors"] = table_df["authors"].apply(
            lambda a: ", ".join(a[:2]) + (" et al." if len(a) > 2 else "")
        )
        table_df["topics"] = table_df["topics"].apply(lambda t: ", ".join(t[:3]))
        table_df = table_df.rename(columns={
            "title":   "Título",
            "authors": "Autores",
            "year":    "Año",
            "venue":   "Venue",
            "topics":  "Tópicos",
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)
