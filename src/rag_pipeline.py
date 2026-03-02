# -*- coding: utf-8 -*-
"""
Research Copilot – RAG Pipeline Orchestrator

Usage:
    python src/rag_pipeline.py          # Runs full ingestion pipeline
    from src.rag_pipeline import query  # Use from app/
"""

import os
import sys
import json
import time

# Allow running directly from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from src.ingestion.pdf_extractor import extract_pdf
from src.ingestion.text_cleaner  import clean_text
from src.chunking.chunker        import chunk_text, CHUNK_CONFIGS
from src.embedding.embedder      import Embedder
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever      import retrieve
from src.generation.generator     import STRATEGIES

PAPERS_DIR   = os.path.join(BASE_DIR, "papers")
CATALOG_PATH = os.path.join(PAPERS_DIR, "paper_catalog.json")
CHROMA_PATH  = os.path.join(BASE_DIR, "chroma_db")

_embedder: Embedder = None
_stores: dict[str, ChromaStore] = {}


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_store(chunk_config: str) -> ChromaStore:
    if chunk_config not in _stores:
        _stores[chunk_config] = ChromaStore(
            path=CHROMA_PATH,
            collection_name=f"research_copilot_{chunk_config}",
        )
    return _stores[chunk_config]


def load_catalog() -> list[dict]:
    """Load paper_catalog.json."""
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Catalog not found: {CATALOG_PATH}")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["papers"]


def build_pipeline(force: bool = False) -> None:
    """
    Full ingestion pipeline:
      1. Read paper_catalog.json
      2. Extract + clean each PDF
      3. Chunk with both configs (small + large)
      4. Embed all chunks
      5. Store in ChromaDB (one collection per config)

    If collections already have data, skips unless force=True.
    """
    # Check if already built
    if not force:
        store_small = _get_store("small")
        if store_small.count() > 0:
            print(f"ChromaDB already has {store_small.count()} chunks (small). Skipping ingestion.")
            return

    papers = load_catalog()
    embedder = _get_embedder()

    print(f"Starting ingestion of {len(papers)} papers...")

    for config in CHUNK_CONFIGS:
        store = _get_store(config)
        if not force and store.count() > 0:
            print(f"  [{config}] Already has {store.count()} chunks. Skipping.")
            continue

        all_chunks: list[dict] = []

        for paper in papers:
            pdf_filename = paper.get("filename", "")
            pdf_path     = os.path.join(PAPERS_DIR, pdf_filename)

            if not os.path.exists(pdf_path):
                print(f"  Warning: PDF not found: {pdf_path}")
                continue

            # Extract
            extraction = extract_pdf(pdf_path)
            raw_text   = extraction["text"]
            if extraction["extraction_warnings"]:
                for w in extraction["extraction_warnings"]:
                    print(f"    [Warning] {pdf_filename}: {w}")

            # Clean
            clean = clean_text(raw_text)
            if not clean.strip():
                print(f"  Warning: Empty text after cleaning for {pdf_filename}")
                continue

            # Build metadata for chunks
            meta = {
                "paper_id":    paper.get("id", ""),
                "paper_title": paper.get("title", ""),
                "authors":     ", ".join(paper.get("authors", [])),
                "year":        int(paper.get("year", 0)),
                "page_number": 0,
                "section":     "",
                "venue":       paper.get("venue", ""),
                "doi":         paper.get("doi", ""),
            }

            # Chunk
            chunks = chunk_text(clean, meta, config=config)
            all_chunks.extend(chunks)
            print(f"  [{config}] {paper['id']}: {len(chunks)} chunks from {pdf_filename}")

        if not all_chunks:
            print(f"  [{config}] No chunks generated. Skipping store.")
            continue

        # Embed
        print(f"  [{config}] Embedding {len(all_chunks)} chunks...")
        texts      = [c["text"] for c in all_chunks]
        embeddings = embedder.embed_texts(texts)

        # Store
        store.add_chunks(all_chunks, embeddings)
        print(f"  [{config}] Stored {store.count()} total chunks in ChromaDB.")

    print("\nPipeline complete.")


def query(
    question: str,
    strategy: str = "Delimitadores",
    n: int = 3,
    chunk_config: str = "small",
) -> dict:
    """
    Execute a RAG query.

    Args:
        question     – the user's research question
        strategy     – one of STRATEGIES keys
        n            – number of chunks to retrieve
        chunk_config – "small" or "large"

    Returns dict with:
        answer, citations, chunks_used, strategy, latency_s, tokens_used
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(STRATEGIES)}")

    t0 = time.time()

    # Retrieve
    chunks, sources = retrieve(question, n_results=n, chunk_config=chunk_config)

    if not chunks:
        return {
            "answer":      "No se encontraron documentos relevantes para esta pregunta.",
            "citations":   [],
            "chunks_used": [],
            "strategy":    strategy,
            "latency_s":   round(time.time() - t0, 2),
            "tokens_used": 0,
        }

    # Generate
    fn = STRATEGIES[strategy]
    answer, citations = fn(question, chunks)

    latency = round(time.time() - t0, 2)

    return {
        "answer":      answer,
        "citations":   citations,
        "chunks_used": chunks,
        "strategy":    strategy,
        "latency_s":   latency,
        "tokens_used": 0,  # can be enhanced with usage tracking
    }


if __name__ == "__main__":
    print("=== Research Copilot – Pipeline RAG ===\n")
    build_pipeline()
    print("\nTest query...")
    result = query("¿Cómo afecta el cambio climático a la seguridad alimentaria?")
    print(f"\nAnswer ({result['strategy']}):\n{result['answer']}")
    print(f"\nCitations:")
    for c in result["citations"]:
        print(f"  {c}")
    print(f"\nLatency: {result['latency_s']}s")
