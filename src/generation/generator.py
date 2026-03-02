# -*- coding: utf-8 -*-
"""
Generation layer: 4 prompting strategies, each reading its template from prompts/.
"""

import os
import json
from typing import Optional
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPTS_DIR  = os.path.join(BASE_DIR, "prompts")

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=key)
    return _client


def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_apa_inline(chunk: dict) -> str:
    """Format a chunk's metadata as an APA in-text citation string."""
    authors_raw = chunk.get("authors", "")
    year        = chunk.get("year", "s.f.")
    title       = chunk.get("paper_title", "")

    # authors may be a comma-separated string
    if authors_raw:
        parts = [a.strip() for a in str(authors_raw).split(",") if a.strip()]
        if len(parts) == 0:
            author_str = "Autor desconocido"
        elif len(parts) == 1:
            author_str = parts[0]
        elif len(parts) == 2:
            author_str = f"{parts[0]} y {parts[1]}"
        else:
            author_str = f"{parts[0]} et al."
    else:
        author_str = "Autor desconocido"

    return f"{author_str} ({year}). *{title}*."


def _build_citations(chunks: list[dict]) -> list[str]:
    """Deduplicate and return APA citation strings for all retrieved chunks."""
    seen: set = set()
    cites: list[str] = []
    for c in chunks:
        key = c.get("paper_id", c.get("paper_title", ""))
        if key not in seen:
            seen.add(key)
            cites.append(_format_apa_inline(c))
    return cites


def _chat(messages: list[dict], response_format=None) -> tuple[str, int]:
    client = _get_client()
    kwargs = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "temperature": 0,
    }
    if response_format:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    tokens = resp.usage.total_tokens if resp.usage else 0
    return resp.choices[0].message.content, tokens


# ─────────────────────────────────────────────
# Strategy 1 – Delimitadores
# ─────────────────────────────────────────────

def strategy_delimitadores(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    template = _load_prompt("v1_delimiters.txt")
    context  = "\n\n".join(c["text"] for c in chunks)
    prompt   = template.format(context=context, question=question)
    answer, _ = _chat([{"role": "user", "content": prompt}])
    return answer, _build_citations(chunks)


# ─────────────────────────────────────────────
# Strategy 2 – JSON Estructurado
# ─────────────────────────────────────────────

def strategy_json(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    template = _load_prompt("v2_json_output.txt")
    context  = "\n\n".join(c["text"] for c in chunks)
    prompt   = template.format(context=context, question=question)
    raw, _ = _chat(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    # Pretty-print so Streamlit can render it
    try:
        parsed  = json.loads(raw)
        answer  = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        answer = raw
    return answer, _build_citations(chunks)


# ─────────────────────────────────────────────
# Strategy 3 – Few-Shot
# ─────────────────────────────────────────────

def strategy_few_shot(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    template = _load_prompt("v3_few_shot.txt")
    context  = "\n\n".join(c["text"] for c in chunks)
    prompt   = template.format(context=context, question=question)
    answer, _ = _chat([{"role": "user", "content": prompt}])
    return answer, _build_citations(chunks)


# ─────────────────────────────────────────────
# Strategy 4 – Chain-of-Thought
# ─────────────────────────────────────────────

def strategy_cot(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    template = _load_prompt("v4_chain_of_thought.txt")
    context  = "\n\n".join(c["text"] for c in chunks)
    prompt   = template.format(context=context, question=question)
    answer, _ = _chat([{"role": "user", "content": prompt}])
    return answer, _build_citations(chunks)


# ─────────────────────────────────────────────
# Public registry
# ─────────────────────────────────────────────

STRATEGIES: dict[str, callable] = {
    "Delimitadores":     strategy_delimitadores,
    "JSON Estructurado": strategy_json,
    "Few-Shot":          strategy_few_shot,
    "Chain-of-Thought":  strategy_cot,
}
