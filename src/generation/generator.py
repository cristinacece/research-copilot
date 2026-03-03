# -*- coding: utf-8 -*-
"""
Generation layer: 4 prompting strategies, each reading its template from prompts/.
"""

import os
import json
import sys
from typing import Optional
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.components.citation import format_apa

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


def _build_citations(chunks: list[dict]) -> list[str]:
    """Deduplicate and return full APA 7 citation strings for all retrieved chunks."""
    seen: set = set()
    cites: list[str] = []
    for c in chunks:
        key = c.get("paper_id", c.get("paper_title", ""))
        if key not in seen:
            seen.add(key)
            cites.append(format_apa({
                "authors":     c.get("authors", ""),
                "year":        c.get("year", "s.f."),
                "paper_title": c.get("paper_title", ""),
                "venue":       c.get("venue", ""),
                "doi":         c.get("doi", ""),
            }))
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

def _build_context(chunks: list[dict]) -> str:
    """Build context string with source labels so the model can cite correctly."""
    parts = []
    for c in chunks:
        authors = c.get("authors", "").strip()
        year    = c.get("year", "")
        # Extract last name(s) for inline citation: take last token of first author
        if authors:
            first_author = authors.split(",")[0].strip()
            last_name    = first_author.split()[-1] if first_author.split() else first_author
            label = f"[Fuente: {last_name}, {year}]\n"
        else:
            label = ""
        parts.append(f"{label}{c['text']}")
    return "\n\n".join(parts)


def strategy_delimitadores(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    template = _load_prompt("v1_delimiters.txt")
    context  = _build_context(chunks)
    prompt   = template.format(context=context, question=question)
    answer, _ = _chat([{"role": "user", "content": prompt}])
    return answer, _build_citations(chunks)


# ─────────────────────────────────────────────
# Strategy 2 – JSON Estructurado
# ─────────────────────────────────────────────

def strategy_json(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    template = _load_prompt("v2_json_output.txt")
    context  = _build_context(chunks)
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
    context  = _build_context(chunks)
    prompt   = template.format(context=context, question=question)
    answer, _ = _chat([{"role": "user", "content": prompt}])
    return answer, _build_citations(chunks)


# ─────────────────────────────────────────────
# Strategy 4 – Chain-of-Thought
# ─────────────────────────────────────────────

def strategy_cot(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    template = _load_prompt("v4_chain_of_thought.txt")
    context  = _build_context(chunks)
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
