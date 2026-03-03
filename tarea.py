# -*- coding: utf-8 -*-
"""
Research Copilot – Pipeline RAG para papers académicos
Seguridad alimentaria, comercio internacional y soberanía alimentaria

Uso:
    python tarea.py          # Ejecuta el pipeline completo
    import tarea             # Para usar desde app.py
"""

import os
import json
import re
import time
import fitz  # PyMuPDF
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from getpass import getpass
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RUTA_PAPERS  = os.path.join(BASE_DIR, "papers")
ARCHIVO_JSON = os.path.join(BASE_DIR, "paper_catalog.json")
CHROMA_PATH  = os.path.join(BASE_DIR, "chroma_db")
MODEL_NAME   = "gpt-4o-mini"
EMBED_MODEL  = "text-embedding-3-small"

# ─────────────────────────────────────────────
# CLIENTES (inicialización lazy)
# ─────────────────────────────────────────────

_client    = None  # OpenAI client
_coleccion = None  # ChromaDB collection


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        key = getpass("Introduce tu API Key de OpenAI: ")
        os.environ["OPENAI_API_KEY"] = key
    return key


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_api_key())
    return _client


def get_collection():
    """Retorna la colección ChromaDB (con caché en proceso)."""
    global _coleccion
    if _coleccion is not None:
        return _coleccion

    api_key = get_api_key()
    cliente_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=EMBED_MODEL
    )
    _coleccion = cliente_chroma.get_or_create_collection(
        name="seguridad_alimentaria", embedding_function=openai_ef
    )
    return _coleccion


# ─────────────────────────────────────────────
# PASO 1: GENERACIÓN DE CATÁLOGO
# ─────────────────────────────────────────────

def get_completion(prompt: str, model: str = MODEL_NAME) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def generate_catalog():
    """Genera paper_catalog.json usando GPT. Si ya existe, lo omite."""
    if os.path.exists(ARCHIVO_JSON):
        print(f"Catálogo ya existe → {ARCHIVO_JSON}. Omitiendo generación.")
        return

    archivos_pdf = [f for f in os.listdir(RUTA_PAPERS) if f.endswith(".pdf")]
    catalogo = {"papers": []}

    print(f"Se encontraron {len(archivos_pdf)} PDFs. Generando catálogo...\n")

    for i, archivo in enumerate(archivos_pdf):
        ruta_completa = os.path.join(RUTA_PAPERS, archivo)
        print(f"Procesando ({i+1}/{len(archivos_pdf)}): {archivo}")
        numero_paginas = 0
        try:
            doc = fitz.open(ruta_completa)
            numero_paginas = doc.page_count
            texto_inicial = ""
            for num_pag in range(min(7, numero_paginas)):
                texto_inicial += doc[num_pag].get_text()
            doc.close()

            if len(texto_inicial.strip()) < 50:
                raise ValueError("PDF sin texto legible (posiblemente escaneado).")

            titulo_ia  = get_completion(f"Extrae el título literal. Responde SOLO con el título.\n###{texto_inicial}###")
            resumen_ia = get_completion(f"Resume en exactamente tres oraciones.\n###{texto_inicial}###")
            year_ia    = get_completion(f"Extrae SOLO el año de publicación de 4 dígitos.\n###{texto_inicial}###")
            autores_ia = get_completion(f"Extrae los autores separados por comas, sin texto extra.\n###{texto_inicial}###")
            topics_ia  = get_completion(f"Determina 5 temas. Lista numerada, 1-2 palabras cada uno.\n###{texto_inicial}###")

            match_year = re.search(r"(19|20)\d{2}", year_ia)
            year_num   = int(match_year.group()) if match_year else 2022
            lista_autores = [a.strip() for a in autores_ia.split(",") if a.strip()]
            lista_topics  = [re.sub(r"^\d+\.\s*", "", t).strip() for t in topics_ia.split("\n") if t.strip()]

            print(f"  ✅ {titulo_ia[:50]}...")

        except Exception as e:
            print(f"  ❌ Error en {archivo}: {e}")
            titulo_ia     = archivo.replace(".pdf", "")
            resumen_ia    = "Error de procesamiento."
            year_num      = 2022
            lista_autores = ["Autor Desconocido"]
            lista_topics  = ["Sin clasificar"]

        catalogo["papers"].append({
            "id":       f"paper_{i+1:03d}",
            "title":    titulo_ia,
            "authors":  lista_autores,
            "year":     year_num,
            "venue":    "",
            "doi":      "",
            "filename": archivo,
            "pages":    numero_paginas,
            "topics":   lista_topics,
            "abstract": resumen_ia,
        })
        time.sleep(2)  # Pausa para no saturar la API

    with open(ARCHIVO_JSON, "w", encoding="utf-8") as f:
        json.dump(catalogo, f, indent=4, ensure_ascii=False)
    print(f"\nCatálogo guardado en {ARCHIVO_JSON}")


# ─────────────────────────────────────────────
# PASO 2: EXTRACCIÓN DE TEXTO DE LOS PDFs
# ─────────────────────────────────────────────

def extract_texts():
    """Extrae texto completo de cada PDF a un archivo _texto.txt."""
    archivos_pdf = [f for f in os.listdir(RUTA_PAPERS) if f.endswith(".pdf")]
    print(f"Extrayendo texto de {len(archivos_pdf)} PDFs...\n")

    for archivo in archivos_pdf:
        nombre_txt = archivo.replace(".pdf", "_texto.txt")
        ruta_txt   = os.path.join(RUTA_PAPERS, nombre_txt)

        if os.path.exists(ruta_txt):
            print(f"  Ya existe: {nombre_txt}")
            continue

        ruta_completa = os.path.join(RUTA_PAPERS, archivo)
        try:
            doc = fitz.open(ruta_completa)
            texto_completo = ""
            for pagina in doc:
                texto_completo += pagina.get_text() + "\n"
            doc.close()

            with open(ruta_txt, "w", encoding="utf-8") as f:
                f.write(texto_completo)
            print(f"  ✅ {archivo} → {nombre_txt}")

        except Exception as e:
            print(f"  ❌ Error en {archivo}: {e}")

    print("\nExtracción completada.")


# ─────────────────────────────────────────────
# PASO 3: VECTORIZACIÓN Y CHROMADB PERSISTENTE
# ─────────────────────────────────────────────

def build_chromadb():
    """Vectoriza los textos e ingesta en ChromaDB. Omite si ya está construida."""
    coleccion = get_collection()

    if coleccion.count() > 0:
        print(f"ChromaDB ya contiene {coleccion.count()} fragmentos. Omitiendo vectorización.")
        return

    archivos_txt = [f for f in os.listdir(RUTA_PAPERS) if f.endswith("_texto.txt")]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documentos, metadatos, ids = [], [], []

    print("Fragmentando textos...")
    for i, archivo in enumerate(archivos_txt):
        with open(os.path.join(RUTA_PAPERS, archivo), "r", encoding="utf-8") as f:
            texto = f.read()
        chunks = text_splitter.split_text(texto)
        nombre_pdf = archivo.replace("_texto.txt", ".pdf")
        for j, chunk in enumerate(chunks):
            documentos.append(chunk)
            metadatos.append({"source": nombre_pdf, "chunk_id": j})
            ids.append(f"doc_{i}_chunk_{j}")

    print(f"Total fragmentos: {len(documentos)}. Vectorizando por lotes de 150...")

    tamaño_lote = 150
    for i in range(0, len(documentos), tamaño_lote):
        fin = min(i + tamaño_lote, len(documentos))
        coleccion.add(
            documents=documentos[i:fin],
            metadatas=metadatos[i:fin],
            ids=ids[i:fin],
        )
        print(f"  → Lote {i}–{fin-1} procesado.")

    print(f"\n✅ ChromaDB construida con {coleccion.count()} fragmentos en {CHROMA_PATH}")


# ─────────────────────────────────────────────
# CATÁLOGO HELPERS PARA CITAS APA
# ─────────────────────────────────────────────

_catalog_map = None


def _get_catalog_map() -> dict:
    """Carga el catálogo y devuelve un dict filename → paper_data (con caché)."""
    global _catalog_map
    if _catalog_map is not None:
        return _catalog_map
    if not os.path.exists(ARCHIVO_JSON):
        return {}
    with open(ARCHIVO_JSON, "r", encoding="utf-8") as f:
        papers = json.load(f)["papers"]
    _catalog_map = {p["filename"]: p for p in papers}
    return _catalog_map


def _format_apa(paper: dict) -> str:
    """Formatea una referencia APA en markdown: **autores** (año). Título. *Venue*. [DOI](url)"""
    authors = paper.get("authors", [])
    year    = paper.get("year", "s.f.")
    title   = paper.get("title", "Sin título")
    venue   = paper.get("venue", "").strip()
    doi     = paper.get("doi", "").strip()

    def fmt_author(a: str) -> str:
        """Convierte 'Nombre Apellido(s)' → 'Apellido(s), N.' para APA."""
        parts = a.strip().split()
        if len(parts) >= 2:
            first = parts[0]            # primera palabra = nombre
            last  = " ".join(parts[1:]) # resto = apellido(s)
            return f"{last}, {first[0]}."
        return a.strip()

    if len(authors) == 0:
        authors_str = "Autor desconocido"
    elif len(authors) == 1:
        authors_str = fmt_author(authors[0])
    elif len(authors) == 2:
        authors_str = f"{fmt_author(authors[0])}, & {fmt_author(authors[1])}"
    else:
        formatted = [fmt_author(a) for a in authors]
        authors_str = ", ".join(formatted[:-1]) + f", & {formatted[-1]}"

    apa = f"**{authors_str}** ({year}). {title}."
    if venue:
        apa += f" *{venue}*."
    if doi:
        url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        apa += f" [DOI/URL]({url})"
    return apa


# ─────────────────────────────────────────────
# RETRIEVAL CON CITAS APA
# ─────────────────────────────────────────────

def recuperar_contexto(pregunta: str, n_resultados: int = 3):
    """Recupera los n_resultados chunks más relevantes y devuelve citas APA."""
    coleccion  = get_collection()
    resultados = coleccion.query(query_texts=[pregunta], n_results=n_resultados)
    contexto   = "\n\n".join(resultados["documents"][0])
    filenames  = list(set(meta["source"] for meta in resultados["metadatas"][0]))

    catalog = _get_catalog_map()
    fuentes_apa = []
    for fn in filenames:
        if fn in catalog:
            fuentes_apa.append(_format_apa(catalog[fn]))
        else:
            fuentes_apa.append(f"`{fn}`")

    return contexto, fuentes_apa


# ─────────────────────────────────────────────
# SYSTEM PROMPTS (estructura recomendada por el profesor)
# ─────────────────────────────────────────────

SYSTEM_PROMPT_V1 = (
    "You are Research Copilot, an expert academic assistant specializing in food "
    "security, international trade, climate change, and political economy.\n\n"
    "YOUR TASK:\n"
    "Answer the question using the provided context as your primary evidence base. "
    "You may draw on your broader academic and expert knowledge to enrich explanations, "
    "fill conceptual gaps, and draw non-linear connections across topics and disciplines.\n\n"
    "RULES:\n"
    "1. Prioritize information from the provided context, but never refuse to answer — "
    "always build the most complete response possible\n"
    "2. When the context offers partial information, synthesize it with your expert "
    "knowledge to deliver a thorough answer\n"
    "3. Use only inline parenthetical citations, e.g. (Author, Year). "
    "DO NOT add a References or Bibliography section — the system appends it automatically\n"
    "4. Be precise and academic in your tone\n"
    "5. Make explicit connections between ideas, even across different papers or "
    "concepts not directly mentioned in the context\n\n"
    "CONTEXT:\n"
    "###\n"
    "{context}\n"
    "###\n\n"
    "USER QUESTION: {question}\n\n"
    "YOUR ANSWER:"
)

SYSTEM_PROMPT_V2 = (
    "You are Research Copilot, an expert academic assistant in food security, "
    "international trade, and political economy.\n\n"
    "Answer the question drawing primarily from the provided context, but supplement "
    "with your expert knowledge to build complete, insightful answers. Make non-linear "
    "connections between concepts when relevant. Always provide a substantive answer — "
    "never leave the answer field empty or say you cannot answer.\n\n"
    "Use confidence levels as follows:\n"
    "- 'high': answer is directly supported by the context\n"
    "- 'medium': answer combines context with broader expert knowledge\n"
    "- 'low': answer draws mainly on general expertise beyond the context\n\n"
    "For citations inside 'answer', use only inline parenthetical format (Author, Year). "
    "For the 'citations' array, use only papers that appear in the provided context — "
    "do NOT invent papers from general knowledge.\n\n"
    "You must respond in the following JSON format:\n\n"
    "{\n"
    '    "answer": "Your detailed answer here",\n'
    '    "confidence": "high|medium|low",\n'
    '    "citations": [\n'
    "        {\n"
    '            "paper": "Paper title",\n'
    '            "authors": "Author names",\n'
    '            "year": 2023,\n'
    '            "quote": "Relevant quote from paper"\n'
    "        }\n"
    "    ],\n"
    '    "related_topics": ["topic1", "topic2"]\n'
    "}\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION: {question}"
)

SYSTEM_PROMPT_V3 = (
    "You are Research Copilot, an expert in food security, international trade, "
    "climate change, and political economy. You build comprehensive, insightful "
    "answers by integrating evidence from the provided papers with your broader "
    "academic knowledge. You always answer fully — you never refuse a question.\n\n"
    "Here are examples of how to answer:\n\n"
    "EXAMPLE 1:\n"
    "Question: What is the main contribution of the transformer paper?\n"
    'Context: "We propose a new simple network architecture, the Transformer, '
    'based solely on attention mechanisms..." (Vaswani et al., 2017, p. 1)\n'
    "Answer: The main contribution of the transformer paper is proposing a novel "
    "neural network architecture that relies entirely on attention mechanisms, "
    "eliminating the need for recurrence and convolutions. According to Vaswani et al. "
    '(2017), "We propose a new simple network architecture, the Transformer, based '
    'solely on attention mechanisms" (p. 1).\n\n'
    "EXAMPLE 2:\n"
    "Question: How does BERT handle bidirectional context?\n"
    'Context: "BERT is designed to pre-train deep bidirectional representations by '
    'jointly conditioning on both left and right context in all layers." '
    "(Devlin et al., 2019, p. 2)\n"
    "Answer: BERT handles bidirectional context through its pre-training strategy. "
    'As Devlin et al. (2019) explain, the model "jointly condition[s] on both left '
    'and right context in all layers" (p. 2), allowing it to build deep bidirectional '
    "representations.\n\n"
    "---\n"
    "Now answer the following. Use the context as your primary evidence, but "
    "connect ideas across the corpus and draw on expert knowledge to form a complete "
    "answer. Use only inline parenthetical citations (Author, Year) — "
    "DO NOT add a References section at the end, the system appends it automatically.\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION: {question}"
)

SYSTEM_PROMPT_V4 = (
    "You are Research Copilot, an expert academic analyst in food security, "
    "international trade, and political economy. You always produce a complete, "
    "well-reasoned answer — you never refuse to answer a question.\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION: {question}\n\n"
    "Think through this step-by-step:\n"
    "1. Identify exactly what the question is asking\n"
    "2. Extract relevant evidence from the context — including indirect or partial evidence\n"
    "3. Draw non-linear connections: link concepts across different papers, "
    "identify underlying mechanisms, and relate to broader academic frameworks\n"
    "4. Supplement with your expert knowledge to fill any gaps in the context\n"
    "5. Formulate a comprehensive answer using inline parenthetical citations (Author, Year). "
    "DO NOT add a References or Bibliography section — the system appends it automatically\n\n"
    "STEP-BY-STEP REASONING:\n"
    "[Your reasoning here]\n\n"
    "FINAL ANSWER:\n"
    "[Your complete answer with inline citations only]"
)


# ─────────────────────────────────────────────
# LAS 4 ESTRATEGIAS DE PROMPT ENGINEERING
# ─────────────────────────────────────────────

def prompt_delimitadores(pregunta: str, n_resultados: int = 3):
    """Estrategia 1: Clear Instructions with Delimiters (V1)."""
    contexto, fuentes_apa = recuperar_contexto(pregunta, n_resultados)
    prompt = SYSTEM_PROMPT_V1.replace("{context}", contexto).replace("{question}", pregunta)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content, fuentes_apa


def prompt_json(pregunta: str, n_resultados: int = 3):
    """Estrategia 2: JSON Structured Output (V2)."""
    contexto, fuentes_apa = recuperar_contexto(pregunta, n_resultados)
    prompt = SYSTEM_PROMPT_V2.replace("{context}", contexto).replace("{question}", pregunta)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content, fuentes_apa


def prompt_few_shot(pregunta: str, n_resultados: int = 3):
    """Estrategia 3: Few-Shot Learning (V3)."""
    contexto, fuentes_apa = recuperar_contexto(pregunta, n_resultados)
    prompt = SYSTEM_PROMPT_V3.replace("{context}", contexto).replace("{question}", pregunta)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content, fuentes_apa


def prompt_cot(pregunta: str, n_resultados: int = 3):
    """Estrategia 4: Chain-of-Thought Reasoning (V4)."""
    contexto, fuentes_apa = recuperar_contexto(pregunta, n_resultados)
    prompt = SYSTEM_PROMPT_V4.replace("{context}", contexto).replace("{question}", pregunta)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content, fuentes_apa


# Diccionario de estrategias (para iterar en evaluación y Streamlit)
ESTRATEGIAS = {
    "Delimitadores":     prompt_delimitadores,
    "JSON Estructurado": prompt_json,
    "Few-Shot Learning": prompt_few_shot,
    "Chain-of-Thought":  prompt_cot,
}

# ─────────────────────────────────────────────
# EVALUACIÓN COMPARATIVA
# ─────────────────────────────────────────────

def evaluate_strategies(pregunta: str = None):
    """Ejecuta las 4 estrategias y mide latencia."""
    if pregunta is None:
        pregunta = "¿De qué manera influye la dependencia de importaciones en la soberanía alimentaria durante una crisis?"

    print(f"\n🔍 PREGUNTA: '{pregunta}'\n")
    print("=" * 70)

    for nombre, funcion in ESTRATEGIAS.items():
        print(f"▶️  {nombre}")
        t0 = time.time()
        try:
            respuesta, fuentes = funcion(pregunta)
            latencia = time.time() - t0
            print(f"⏱️  Latencia: {latencia:.2f}s")
            print(f"📚 Fuentes: {fuentes}")
            print(f"🤖 Respuesta:\n{respuesta}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
        print("=" * 70)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Research Copilot – Pipeline RAG ===\n")
    generate_catalog()
    extract_texts()
    build_chromadb()
    print("\n✅ Pipeline completo. Ejecutando evaluación de estrategias...\n")
    evaluate_strategies()


if __name__ == "__main__":
    main()