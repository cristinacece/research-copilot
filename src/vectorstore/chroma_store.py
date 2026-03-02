# -*- coding: utf-8 -*-
"""
ChromaDB persistent vector store wrapper.
"""

import os
from typing import Optional
import chromadb

COLLECTION_NAME = "research_copilot"


class ChromaStore:
    """Wraps a ChromaDB PersistentClient with helper methods."""

    def __init__(self, path: str = "chroma_db", collection_name: str = COLLECTION_NAME):
        self._path = path
        self._collection_name = collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=self._path)
            # embedding_function=None porque pasamos embeddings precomputados
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        return self._collection

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """
        Ingest chunks with pre-computed embeddings into ChromaDB.

        Args:
            chunks     – list of chunk dicts (from chunker.chunk_text)
            embeddings – corresponding list of embedding vectors
        """
        if not chunks:
            return

        collection = self._get_collection()

        ids        = [c["chunk_id"] for c in chunks]
        documents  = [c["text"]     for c in chunks]
        metadatas  = []

        for c in chunks:
            meta = c.get("metadata", {})
            # ChromaDB only accepts str/int/float/bool metadata values
            safe_meta = {
                "paper_id":     str(meta.get("paper_id",    "")),
                "paper_title":  str(meta.get("paper_title", "")),
                "authors":      str(meta.get("authors",     "")),
                "year":         int(meta.get("year",         0)),
                "page_number":  int(meta.get("page_number",  0)),
                "section":      str(meta.get("section",     "")),
                "chunk_config": str(meta.get("chunk_config","small")),
                "token_count":  int(c.get("token_count",     0)),
                "venue":        str(meta.get("venue",        "")),
                "doi":          str(meta.get("doi",          "")),
            }
            metadatas.append(safe_meta)

        # Add in batches of 150
        batch_size = 150
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for the closest chunks.

        Returns list of dicts with keys:
            chunk_id, text, similarity_score, + all metadata fields
        """
        collection = self._get_collection()

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        result = collection.query(**kwargs)

        hits = []
        docs      = result["documents"][0]
        metas     = result["metadatas"][0]
        distances = result["distances"][0]
        ids_list  = result["ids"][0]

        for chunk_id, doc, meta, dist in zip(ids_list, docs, metas, distances):
            # Cosine distance → similarity: sim = 1 - dist
            similarity = round(1.0 - float(dist), 4)
            hits.append({
                "chunk_id":        chunk_id,
                "text":            doc,
                "similarity_score": similarity,
                **meta,
            })

        return hits

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._get_collection().count()

    def delete_collection(self) -> None:
        """Drop the entire collection (destructive)."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._path)
        self._client.delete_collection(self._collection_name)
        self._collection = None
