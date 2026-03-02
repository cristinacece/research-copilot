# Research Copilot

**Sistema RAG (Retrieval-Augmented Generation) para análisis académico de papers sobre seguridad alimentaria y comercio internacional.**

> Autora: **Cristina Celeste Tamay Blanco**

---

## Features

- **Pipeline RAG completo**: extracción PDF → limpieza → chunking con tiktoken → embeddings OpenAI → ChromaDB persistente
- **4 estrategias de prompting**: Delimitadores, JSON Estructurado, Few-Shot y Chain-of-Thought
- **Chat multi-turno** con memoria de los últimos 4 intercambios
- **Citas APA 7** generadas automáticamente a partir de los metadatos del corpus
- **Paper Browser** con filtros por año, tópico y autor
- **Analytics**: visualizaciones interactivas del corpus (Plotly)
- **Evaluación comparativa** de estrategias sobre 20 preguntas de referencia

### Screenshots

> *(Ver `demo/screenshots/` para capturas actualizadas)*

---

## Architecture

```
Usuario
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  Streamlit UI  (app/main.py)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────┐ │
│  │   Chat   │  │ Browser  │  │Analytics │  │Cfg  │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────┘ │
└─────────────────────────────────────────────────────┘
           │ query(question, strategy, n, config)
           ▼
┌─────────────────────────────────────────────────────┐
│  RAG Pipeline  (src/rag_pipeline.py)                │
│                                                     │
│  Ingestion          Chunking        Embedding       │
│  pdf_extractor ──►  chunker    ──►  embedder        │
│  text_cleaner        (tiktoken)     (OpenAI)        │
│                                        │            │
│                            ┌───────────┘            │
│                            ▼                        │
│                     ChromaStore                     │
│                     (chroma_db/)                    │
│                            │                        │
│  Retrieval   ◄─────────────┘                        │
│  retriever.py                                       │
│       │                                             │
│       ▼                                             │
│  Generation  (4 strategies)                         │
│  generator.py  ──►  OpenAI GPT-4o-mini              │
└─────────────────────────────────────────────────────┘
```

**Módulos src/**

| Módulo | Responsabilidad |
|--------|----------------|
| `ingestion/pdf_extractor.py` | Extrae texto y metadatos de PDFs con PyMuPDF |
| `ingestion/text_cleaner.py` | Limpia guiones, espacios, números de página |
| `chunking/chunker.py` | Fragmenta texto con conteo exacto de tokens (tiktoken) |
| `embedding/embedder.py` | Genera embeddings en lotes (text-embedding-3-small) |
| `vectorstore/chroma_store.py` | Almacena y recupera vectores con ChromaDB |
| `retrieval/retriever.py` | Orquesta embedding de consulta + búsqueda vectorial |
| `generation/generator.py` | Aplica las 4 estrategias de prompting con GPT-4o-mini |
| `rag_pipeline.py` | Orquestador principal: `build_pipeline()` y `query()` |

---

## Installation

### Prerrequisitos
- Python 3.10+
- OpenAI API Key

### 1. Clonar el repositorio

```bash
git clone https://github.com/cristinacece/research-copilot.git
cd research-copilot
```

### 2. Crear entorno virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API Key

```bash
cp .env.example .env
# Editar .env y agregar tu clave:
# OPENAI_API_KEY=sk-...tu-clave-aquí
```

### 5. Construir el pipeline (primera vez)

```bash
python src/rag_pipeline.py
```

Este comando extrae texto de los 21 PDFs, genera chunks con tiktoken, los embede con OpenAI y los almacena en ChromaDB. Puede tardar varios minutos.

### 6. Lanzar la aplicación

```bash
streamlit run app/main.py
```

---

## Usage

### Chat RAG

Navega a la pestaña **💬 Chat** y escribe tu pregunta. Selecciona la estrategia de prompting y el número de chunks a recuperar.

**Ejemplo de preguntas:**
- *¿Cómo afecta el cambio climático a la seguridad alimentaria?*
- *¿Qué relación existe entre las restricciones comerciales y la soberanía alimentaria?*
- *Sintetiza los principales mecanismos por los cuales el comercio puede amenazar o fortalecer la seguridad alimentaria.*

### Paper Browser

Filtra los 21 papers por año, tópico o autor. Cada tarjeta muestra el abstract y la cita APA generada automáticamente.

### Evaluación

```bash
python eval/evaluate.py --n 3 --config small
```

Genera resultados en `eval/results/eval_YYYYMMDD_HHMMSS.json`.

### Tests

```bash
python -m pytest tests/ -v
```

---

## Technical Details

### Configuraciones de Chunking

| Config | Tamaño (tokens) | Overlap (tokens) | Uso recomendado |
|--------|----------------|-----------------|----------------|
| `small` | 256 | 50 | Mayor precisión, respuestas concretas |
| `large` | 1024 | 100 | Más contexto por chunk, preguntas de síntesis |

Modelo de tokenización: `tiktoken.encoding_for_model("gpt-4o-mini")`

### Estrategias de Prompting

| Estrategia | Descripción | Mejor para |
|-----------|-------------|-----------|
| **Delimitadores** | Contexto enmarcado con `###` | Respuestas directas y factuales |
| **JSON Estructurado** | Respuesta como `{hipotesis, variables, mecanismo}` | Análisis estructurado, exportación de datos |
| **Few-Shot** | Calibrado con ejemplos del marco Keohane & Nye | Respuestas académicamente rigurosas |
| **Chain-of-Thought** | Razonamiento causal en 4 pasos | Preguntas analíticas complejas |

### Modelos utilizados

- **Embeddings**: `text-embedding-3-small` (1536 dimensiones)
- **Generación**: `gpt-4o-mini` (temperatura 0 para reproducibilidad)
- **Vector Store**: ChromaDB con espacio coseno

---

## Evaluation Results

> Ejecuta `python eval/evaluate.py` para generar resultados actualizados.
> Los resultados se guardan en `eval/results/`.

### Métricas de latencia promedio (referencia)

| Estrategia | Latencia promedio |
|-----------|-----------------|
| Delimitadores | ~2-3s |
| JSON Estructurado | ~2-3s |
| Few-Shot | ~3-4s |
| Chain-of-Thought | ~3-5s |

*Latencias varían según carga de la API de OpenAI.*

---

## Limitations

1. **Dependencia de OpenAI**: el sistema requiere conexión a internet y una API key válida. No funciona offline.
2. **PDFs escaneados**: los archivos imagen sin capa OCR no pueden ser procesados por PyMuPDF. Solo se extraen PDFs con texto seleccionable.
3. **Idioma**: el corpus y las estrategias de prompting están optimizados para español. Preguntas en otros idiomas pueden producir resultados de menor calidad.
4. **Contexto limitado**: el modelo `gpt-4o-mini` tiene una ventana de contexto de 128K tokens, pero el pipeline usa solo los top-N chunks recuperados. Preguntas que requieran integrar muchos papers simultáneamente pueden perder información relevante.
5. **Sin verificación de hechos**: el sistema puede generar respuestas plausibles pero incorrectas (*hallucinations*). Las citas APA son automáticas y deben verificarse contra los documentos originales.

---

## Author

**Cristina Celeste Tamay Blanco**
GitHub: [@cristinacece](https://github.com/cristinacece)

---

## License

Este proyecto fue desarrollado con fines académicos.
