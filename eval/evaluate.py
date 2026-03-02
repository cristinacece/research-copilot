# -*- coding: utf-8 -*-
"""
Evaluation script – runs all 4 strategies over 20 questions and saves results.

Usage:
    python eval/evaluate.py
"""

import os
import sys
import json
import time
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from src.rag_pipeline import query as rag_query
from src.generation.generator import STRATEGIES

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def evaluate(n_chunks: int = 3, chunk_config: str = "small") -> dict:
    """Run all strategies on all questions, return full results dict."""
    questions = load_questions()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp":    timestamp,
        "chunk_config": chunk_config,
        "n_chunks":     n_chunks,
        "strategies":   list(STRATEGIES.keys()),
        "evaluations":  [],
    }

    total = len(questions) * len(STRATEGIES)
    done  = 0

    for q in questions:
        qid  = q["id"]
        qtyp = q["type"]
        qtxt = q["question"]
        print(f"\n[{qid}] ({qtyp}) {qtxt[:70]}...")

        q_results = {"id": qid, "type": qtyp, "question": qtxt, "responses": {}}

        for strategy_name in STRATEGIES:
            print(f"  → {strategy_name}", end=" ", flush=True)
            t0 = time.time()
            try:
                result = rag_query(
                    question=qtxt,
                    strategy=strategy_name,
                    n=n_chunks,
                    chunk_config=chunk_config,
                )
                latency = round(time.time() - t0, 2)
                q_results["responses"][strategy_name] = {
                    "answer":    result["answer"],
                    "citations": result["citations"],
                    "latency_s": latency,
                    "success":   True,
                }
                print(f"✅ ({latency}s)")
            except Exception as exc:
                latency = round(time.time() - t0, 2)
                q_results["responses"][strategy_name] = {
                    "answer":    "",
                    "citations": [],
                    "latency_s": latency,
                    "success":   False,
                    "error":     str(exc),
                }
                print(f"❌ {exc}")

            done += 1

        results["evaluations"].append(q_results)

    # Save full results
    out_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to {out_path}")

    # Print summary
    print("\n=== LATENCY SUMMARY ===")
    for strategy_name in STRATEGIES:
        latencies = [
            r["responses"].get(strategy_name, {}).get("latency_s", 0)
            for r in results["evaluations"]
        ]
        avg = round(sum(latencies) / len(latencies), 2) if latencies else 0
        print(f"  {strategy_name}: avg {avg}s")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG strategies")
    parser.add_argument("--n",      type=int, default=3,       help="Chunks to retrieve per query")
    parser.add_argument("--config", type=str, default="small", help="Chunk config: small or large")
    args = parser.parse_args()

    print(f"Starting evaluation: {len(list(STRATEGIES))} strategies × 20 questions")
    print(f"Config: n={args.n}, chunk_config={args.config}\n")
    evaluate(n_chunks=args.n, chunk_config=args.config)
