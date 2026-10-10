"""DeepEval-based evaluation runner: orchestrates the 4 eval categories
(retriever, generator, pipeline, application) against the real
pipeline, in either offline (deterministic, no API key) or live (real
LLM + real DeepEval judge) mode.

One shared golden dataset (golden_dataset.py, built once from
test_cases.json + the independent GoldCorpus) and one shared set of
real `answer_query()` results feed all 4 categories -- generation is
never re-run per category. `--category` lets any subset run on its
own; a category not requested is simply absent from the report (and
its quality gates are skipped, not failed -- see quality_gates.py).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from src.rag.config import RETRIEVAL_DOCUMENTS_PATH
from src.rag.generation import answer_query
from src.rag.llm_provider import LLMConfig, OpenAIStructuredClient
from src.rag.retrieve import RetrievalEngine

from . import application_eval, generator_eval, pipeline_eval, retriever_eval
from .gold_labels import load_gold_corpus
from .golden_dataset import build_golden_dataset
from .judge import judge_model
from .quality_gates import QualityGates, evaluate_gates
from .schema import load_test_suite

ALL_CATEGORIES = ("retriever", "generator", "pipeline", "application")
_NEEDS_ANSWER_QUERY = {"pipeline", "application"}


def corpus_fingerprint() -> dict:
    """Identity of the retrieval corpus this run measured."""
    try:
        data = RETRIEVAL_DOCUMENTS_PATH.read_bytes()
    except OSError:
        return {"retrieval_documents": None}
    modified = datetime.fromtimestamp(RETRIEVAL_DOCUMENTS_PATH.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    return {
        "retrieval_documents": len(json.loads(data)),
        "retrieval_documents_sha256": hashlib.sha256(data).hexdigest()[:12],
        "retrieval_documents_modified": modified,
    }


def run_deepeval_suite(
    mode: str = "offline",
    test_cases_path: str = "data/evaluation/test_cases.json",
    gates_path=None,
    categories: Optional[list[str]] = None,
) -> dict:
    assert mode in ("offline", "live")
    categories = list(categories) if categories else list(ALL_CATEGORIES)
    unknown = set(categories) - set(ALL_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown eval categories: {sorted(unknown)}")

    suite = load_test_suite(test_cases_path)
    corpus = load_gold_corpus()
    engine = RetrievalEngine.build()
    goldens = build_golden_dataset(suite.cases, corpus)

    llm_client = OpenAIStructuredClient() if mode == "live" else None
    config = LLMConfig.from_env()

    answers: dict = {}
    usage_by_id: dict = {}
    latency_by_id: dict = {}
    if _NEEDS_ANSWER_QUERY & set(categories):
        for g in goldens:
            calls_before = llm_client.call_count if llm_client else 0
            answers[g.id] = answer_query(g.question, engine, llm_client=llm_client, no_llm=(mode == "offline"))
            # llm_client persists across every case in a --live run; only record usage/latency
            # when THIS case actually made a new real call, so a case that short-circuited
            # before generation never misreports a PREVIOUS case's usage as its own.
            if llm_client is not None and llm_client.call_count > calls_before:
                usage_by_id[g.id] = llm_client.last_usage
                latency_by_id[g.id] = llm_client.last_latency_seconds

    evidence_by_id = {d.evidence_id: d for d in engine.retrieval_documents}

    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "judge_model": judge_model(),
        "categories_run": categories,
        "corpus": {"product_count": len(corpus.products), "evidence_count": len(corpus.evidence), **corpus_fingerprint()},
        "model_configuration": {
            "provider": config.provider, "model": config.model, "temperature": config.temperature,
            "api_key_configured": config.available if mode == "live" else None,
        },
    }

    if "retriever" in categories:
        report["retriever"] = retriever_eval.run(goldens, engine, mode)
    if "generator" in categories:
        report["generator"] = generator_eval.run(goldens, engine, llm_client, mode)
    if "pipeline" in categories:
        report["pipeline"] = pipeline_eval.run(goldens, answers, mode)
    if "application" in categories:
        report["application"] = application_eval.run(
            goldens, answers, engine, corpus, evidence_by_id, mode, usage_by_id, latency_by_id,
        )

    gates = QualityGates.load(gates_path) if gates_path else QualityGates.load()
    gate_results = evaluate_gates(gates, report)
    report["quality_gates"] = [
        {"name": g.name, "threshold": g.threshold, "actual": g.actual, "passed": g.passed, "comparison": g.comparison}
        for g in gate_results
    ]
    report["quality_gates_all_passed"] = all(g.passed for g in gate_results)
    return report
