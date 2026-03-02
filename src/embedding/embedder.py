# -*- coding: utf-8 -*-
"""
OpenAI embedding wrapper with lazy client initialization.
"""

import os
from typing import Optional
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 150


class Embedder:
    """Lazy-initialized OpenAI embedder."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key."
                )
            self._client = OpenAI(api_key=key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches of BATCH_SIZE."""
        client = self._get_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = client.embeddings.create(model=EMBED_MODEL, input=batch)
            # response.data is sorted by index
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]
