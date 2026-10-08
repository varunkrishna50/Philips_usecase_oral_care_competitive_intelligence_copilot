"""3C.4: reuses the existing 3A retrieval ablation
(scripts/evaluate_retrieval.py) rather than duplicating gold-case
construction or Recall@K logic. That script's own gold cases are built
from `RetrievalEngine`'s product/evidence corpora (acceptable there --
it is 3A's OWN pre-existing evaluation of retrieval mechanics, not the
3C.1 test_cases.json gold labels, which come from gold_labels.py
instead per the 3C.1 integrity rule).

If the embedding model can't load (offline machine, no cached
weights), semantic/hybrid/hybrid+reranked are skipped and reported as
such rather than crashing the whole harness.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def run_retrieval_ablation(engine) -> dict:
    """Returns {"status": "ok", "total_cases": int, "ablation": {...}}
    or {"status": "skipped", "reason": "..."} if the embedding model
    is unavailable."""
    import evaluate_retrieval as ev  # noqa: E402  (scripts/evaluate_retrieval.py)

    cases = ev.build_gold_cases(engine)
    if not cases:
        return {"status": "skipped", "reason": "no gold cases could be built from the current corpus", "total_cases": 0}

    try:
        # Touch the embedding index once up front so a model-load failure
        # is caught here, not mid-ablation after BM25 has already run.
        engine.embedding_index.search(cases[0].query, top_k=1)
    except Exception as exc:  # noqa: BLE001 -- any load/runtime failure means "skip semantic/hybrid"
        return {
            "status": "skipped",
            "reason": f"embedding model unavailable: {exc}",
            "total_cases": len(cases),
        }

    ablation, total = ev.run_ablation(engine, cases)
    return {"status": "ok", "total_cases": total, "ablation": ablation}


def run_behavioral_checks(engine) -> dict:
    """Reuses 3A's existing exhaustive/not-in-corpus/ambiguous/injection
    checks (scripts/evaluate_retrieval.py) -- these overlap in spirit
    with the 3C.1 test_cases.json safety/exhaustive cases but operate
    directly on the retrieval layer (no generation), so keeping them
    here avoids losing 3A's own pre-existing coverage."""
    import evaluate_retrieval as ev

    checks = {
        "exhaustive_pressure_sensing": ev.check_exhaustive(engine),
        "not_in_corpus_refusal": ev.check_not_in_corpus(engine),
        "ambiguous_clarification": ev.check_ambiguous(engine),
        "user_prompt_injection": ev.check_user_prompt_injection(engine),
        "poisoned_evidence_isolation": ev.check_poisoned_evidence(engine),
    }
    return {name: {"passed": ok, "detail": detail} for name, (ok, detail) in checks.items()}
