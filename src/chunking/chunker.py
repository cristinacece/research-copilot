# -*- coding: utf-8 -*-
"""
Token-aware text chunking using tiktoken.

Two built-in configurations:
    "small"  – 256 tokens, 50-token overlap
    "large"  – 1024 tokens, 100-token overlap
"""

from typing import Optional
import tiktoken

CHUNK_CONFIGS: dict[str, dict] = {
    "small": {"size": 256,  "overlap": 50},
    "large": {"size": 1024, "overlap": 100},
}

_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str = "gpt-4o-mini") -> tiktoken.Encoding:
    if model not in _ENCODING_CACHE:
        _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
    return _ENCODING_CACHE[model]


def chunk_text(
    text: str,
    metadata: dict,
    config: str = "small",
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    Split *text* into overlapping token-based chunks.

    Args:
        text     – the cleaned document text
        metadata – arbitrary dict attached to every chunk (e.g. paper_id, year)
        config   – "small" (256 tok) or "large" (1024 tok)
        model    – tiktoken model name used for encoding

    Returns:
        List of chunk dicts:
            chunk_id     – "{paper_id}_c{n}_{config}"
            text         – the chunk text string
            token_count  – exact token count
            start_token  – token offset in document
            end_token    – token offset (exclusive)
            metadata     – copy of metadata dict + chunk_config
    """
    if config not in CHUNK_CONFIGS:
        raise ValueError(f"Unknown chunk config '{config}'. Choose from: {list(CHUNK_CONFIGS)}")

    cfg = CHUNK_CONFIGS[config]
    size: int    = cfg["size"]
    overlap: int = cfg["overlap"]
    stride: int  = size - overlap

    enc = _get_encoding(model)
    tokens = enc.encode(text)

    if not tokens:
        return []

    paper_id = metadata.get("paper_id", "doc")
    chunks: list[dict] = []
    start = 0

    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        chunk_index = len(chunks)

        chunks.append({
            "chunk_id":    f"{paper_id}_c{chunk_index:04d}_{config}",
            "text":        chunk_text_str,
            "token_count": len(chunk_tokens),
            "start_token": start,
            "end_token":   end,
            "metadata":    {**metadata, "chunk_config": config},
        })

        if end == len(tokens):
            break
        start += stride

    return chunks
