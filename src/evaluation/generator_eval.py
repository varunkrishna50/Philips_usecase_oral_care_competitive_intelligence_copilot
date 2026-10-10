"""Category 2 -- Generator alone: given known-good (golden) context,
does the generator produce a faithful, relevant answer?

Bypasses retrieval entirely: feeds the REAL generation hook
(`_call_llm_for_segments` in src/rag/generation.py) the REAL
RetrievalDocument objects selected by gold evidence_id membership
(golden_dataset.golden_retrieval_documents), not by the retrieval
ranking algorithm -- the same technique the old runner.py's
poisoned-evidence test already used to reach into production code with
a hand-picked document. A retrieval bug can then never be blamed for
(or hide) a generation problem, and vice versa.

Metrics: FaithfulnessMetric, AnswerRelevancyMetric, and
CitationFaithfulnessMetric (deepeval.metrics.community -- the
"citation presence and validity" check: does every [S1]/[S2] marker in
the answer point to the gold passage that actually supports that
specific claim, stricter than Faithfulness alone).

Live-mode only: a deterministic (no-LLM) fallback has no "generator" to
evaluate, so offline-mode cases are reported not_exercised, same
convention as every other LLM-requiring case in this harness.

For `comparison` cases this always exercised the same generation hook
production uses for comparable attributes (`_answer_comparison` in
`src/rag/generation.py` is itself LLM-first now, falling back to
deterministic arithmetic only when the LLM is unavailable/fails) -- so
unlike before, this is no longer a code path production itself never
takes for comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.metrics.community import CitationFaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.rag.citations import assign_markers, render_answer_text, validate_segments
from src.rag.generation import _call_llm_for_segments
from src.rag.retrieve import RetrievalEngine

from .deepeval_support import MetricOutcome, aggregate_scores, measure
from .golden_dataset import GoldenCase, golden_retrieval_documents
from .judge import judge_model

GENERATION_INSTRUCTION = (
    "Answer the user's question using the evidence below. Create one segment per distinct fact, "
    "each citing the evidence handle(s) that support it."
)


@dataclass
class GeneratorCaseResult:
    id: str
    category: str
    exercised: bool
    reason: str
    actual_output: str = ""
    metrics: list[MetricOutcome] = field(default_factory=list)


def evaluate_case(golden: GoldenCase, engine: RetrievalEngine, llm_client, mode: str) -> GeneratorCaseResult:
    if mode != "live":
        return GeneratorCaseResult(golden.id, golden.category, False, "requires --live (a real generator call)")
    docs = golden_retrieval_documents(golden, engine.retrieval_documents)
    if not docs:
        return GeneratorCaseResult(golden.id, golden.category, False, "no golden context for this category")

    evidence_by_id = {d.evidence_id: d for d in docs}
    segments, failure = _call_llm_for_segments(GENERATION_INSTRUCTION, golden.question, docs, llm_client)
    if failure:
        return GeneratorCaseResult(golden.id, golden.category, False, f"generator unavailable: {failure}")

    validated = validate_segments(segments, evidence_by_id, engine.product_corpus)
    markers = assign_markers(validated, evidence_by_id)
    actual_output = render_answer_text(validated, markers) or ""
    if not actual_output.strip():
        return GeneratorCaseResult(golden.id, golden.category, False, "generator produced no usable segments on golden context")

    model = judge_model()
    # The judge must see the SAME evidence representation the generator was
    # actually given (docs' own shown_text, production's rendered form) --
    # not golden.golden_context (GoldCorpus's raw evidence_text, which can be
    # an unparsed structured-data blob). Scoring a claim against a different,
    # messier rendering of the same fact than what produced it makes a fully
    # faithful citation look unverifiable to the judge.
    retrieval_context = [d.shown_text for d in docs]
    test_case = LLMTestCase(input=golden.question, actual_output=actual_output, retrieval_context=retrieval_context)
    metrics = [
        measure(FaithfulnessMetric(model=model), test_case),
        measure(AnswerRelevancyMetric(model=model), test_case),
        measure(CitationFaithfulnessMetric(model=model), test_case, name="CitationFaithfulness"),
    ]
    return GeneratorCaseResult(golden.id, golden.category, True, "generator scored on known-good context",
                                actual_output=actual_output, metrics=metrics)


def run(goldens: list[GoldenCase], engine: RetrievalEngine, llm_client, mode: str) -> dict:
    case_results = [evaluate_case(g, engine, llm_client, mode) for g in goldens]
    all_metrics = [m for c in case_results for m in c.metrics]
    return {
        "judge_model": judge_model(),
        "mode": mode,
        "cases": [
            {
                "id": c.id, "category": c.category, "exercised": c.exercised, "reason": c.reason,
                "actual_output": c.actual_output, "metrics": [m.to_dict() for m in c.metrics],
            }
            for c in case_results
        ],
        "total_cases": len(case_results),
        "exercised_cases": sum(1 for c in case_results if c.exercised),
        "aggregate": aggregate_scores(all_metrics),
    }
