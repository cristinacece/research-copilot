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
# RETRIEVAL
# ─────────────────────────────────────────────

def recuperar_contexto(pregunta: str, n_resultados: int = 3):
    """Recupera los n_resultados chunks más relevantes para la pregunta."""
    coleccion = get_collection()
    resultados = coleccion.query(query_texts=[pregunta], n_results=n_resultados)
    contexto = "\n\n".join(resultados["documents"][0])
    fuentes  = list(set(meta["source"] for meta in resultados["metadatas"][0]))
    return contexto, fuentes


# ─────────────────────────────────────────────
# LAS 4 ESTRATEGIAS DE PROMPT ENGINEERING
# ─────────────────────────────────────────────

def prompt_delimitadores(pregunta: str, n_resultados: int = 3):
    """Estrategia 1: Delimitadores ### para aislar el contexto."""
    contexto, fuentes = recuperar_contexto(pregunta, n_resultados)
    prompt = f"""Eres un analista de relaciones internacionales.
Responde la pregunta utilizando ÚNICAMENTE la información delimitada por ###.

###{contexto}###

Pregunta: {pregunta}"""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content, fuentes


def prompt_json(pregunta: str, n_resultados: int = 3):
    """Estrategia 2: Respuesta estructurada como JSON."""
    contexto, fuentes = recuperar_contexto(pregunta, n_resultados)
    prompt = f"""Analiza el contexto y devuelve OBLIGATORIAMENTE un JSON válido con esta estructura:
{{
    "hipotesis": "resumen del impacto",
    "variables_clave": ["var1", "var2"],
    "mecanismo_transmision": "cómo afecta un factor al otro"
}}

Contexto: {contexto}
Pregunta: {pregunta}"""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content, fuentes


def prompt_few_shot(pregunta: str, n_resultados: int = 3):
    """Estrategia 3: Few-Shot calibrado con marco de interdependencia compleja."""
    contexto, fuentes = recuperar_contexto(pregunta, n_resultados)
    prompt = f"""Responde siguiendo el rigor del marco de la interdependencia compleja.
Ejemplos de calibración:

Pregunta: ¿Cómo la asimetría comercial afecta la disponibilidad de alimentos?
Respuesta: La asimetría en el comercio internacional otorga poder estructural a los países exportadores. Ante un shock exógeno, los estados dependientes sufren disrupciones en la red de suministro, comprometiendo su seguridad alimentaria local (Keohane & Nye, 1977).

Pregunta: ¿Qué rol juegan las organizaciones transnacionales?
Respuesta: Reducen los costos de transacción y actúan como canales múltiples de contacto, mitigando en cierta medida la vulnerabilidad de los estados menos favorecidos (Autor, Año).

Contexto: {contexto}
Pregunta real a resolver: {pregunta}"""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content, fuentes


def prompt_cot(pregunta: str, n_resultados: int = 3):
    """Estrategia 4: Chain-of-Thought con razonamiento causal en 3 pasos."""
    contexto, fuentes = recuperar_contexto(pregunta, n_resultados)
    prompt = f"""Eres un investigador de economía política internacional. Resuelve la consulta paso a paso:
1. Identifica las variables macroeconómicas mencionadas en el contexto.
2. Explica la cadena causal (cómo A afecta a B en términos de redes de suministro).
3. Formula tu conclusión final.

Contexto: {contexto}
Pregunta: {pregunta}"""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content, fuentes


# Diccionario de estrategias (para iterar en evaluación y Streamlit)
ESTRATEGIAS = {
    "Delimitadores":    prompt_delimitadores,
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
