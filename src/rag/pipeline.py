#!/usr/bin/env python3
"""Step 3b: grounded generation CLI.

    python -m src.rag.pipeline --query "..." [--debug] [--no-llm] [--top-k 10] [--alpha 0.5]

Works identically with or without OPENAI_API_KEY set: with no key (or
--no-llm, or any LLM failure), the answer is built entirely from the
deterministic comparison engine / status-aware templates (3B.3) and
clearly says so.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from .generation import answer_query
from .llm_provider import LLMConfig
from .retrieve import RetrievalEngine


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no-llm", action="store_true", help="Force the deterministic fallback, even if an API key is configured.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    engine = RetrievalEngine.load()
    result = answer_query(args.query, engine, top_k=args.top_k, alpha=args.alpha, no_llm=args.no_llm)

    if args.debug:
        config = LLMConfig.from_env()
        print(f"LLM provider={config.provider} model={config.model} temperature={config.temperature} configured={config.available}")
        print(f"used_llm={result.used_llm}")
        rr = result.retrieval_result
        if rr is not None:
            print(f"intent={rr.intent} product_match={rr.product_match.kind} attributes={rr.attributes}")
        print(f"segments: {len(result.segments)} ({sum(1 for s in result.segments if s.valid)} valid)")
        for seg in result.segments:
            status = "VALID" if seg.valid else f"REJECTED ({'; '.join(seg.reasons)})"
            print(f"  [{status}] {seg.text[:100]!r} citing {seg.citation_ids}")
        print()

    print(result.answer_text)
    if result.sources_text:
        print()
        print(result.sources_text)
    if result.llm_unavailable_reason:
        print()
        print(f"[{result.llm_unavailable_reason}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
