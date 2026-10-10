"""Category 1 -- Retriever alone: do the retrieved policy chunks fit the
question and the golden answer?

Calls `RetrievalEngine.retrieve()` directly (contextualization -> hybrid
BM25+semantic retrieval -> rerank -> hard evidence filter), independent
of generation.py's rule-engine short-circuits (medical advice is checked
in generation.py, BEFORE retrieval is even reached in the real app) --
this category measures the retriever on its own, so it always runs a
real retrieve() call for every case that has document-level ground
truth. Metrics: ContextualPrecision, ContextualRecall, ContextualRelevancy
(DeepEval built-ins) -- nothing else.

Cases with no document-level ground truth at all (out_of_corpus, the
generic medical_advice/injection probes) have nothing for
ContextualPrecision/Recall to score against and are reported
not_exercised here -- scoring them against the mechanical
expected_behavior fallback would produce a meaningless number, not a
retrieval signal.

ContextualRelevancyMetric is additionally skipped (not just reported
differently -- never even constructed) for categories where the
correct retrieval behavior is to surface the surrounding context for a
fact that ISN'T on the page (medical_advice, missing_attribute): the
judge only asks "does this context answer the literal question", and
for these categories nothing CAN answer it by design, so the metric
would always penalize correct behavior. ContextualPrecision/Recall
don't have this problem -- they're judged against the golden answer
(which itself says "not mentioned"), not the bare question.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from .deepeval_support import MetricOutcome, aggregate_scores, measure_if_live
from .golden_dataset import GoldenCase
from .judge import judge_model

NO_RELEVANT_EVIDENCE_BY_DESIGN_CATEGORIES = {"medical_advice", "missing_attribute"}


def has_document_scope(golden: GoldenCase) -> bool:
    return bool(golden.test_case.gold_products)


def chunks_from_retrieval_result(result) -> list[str]:
    """Shared with pipeline_eval.py, which reuses the SAME RetrievalResult
    the real answer_query() call already produced instead of retrieving
    twice. Each chunk is prefixed with its product name -- the same
    disambiguation the real LLM prompt gets (generation.py's own
    evidence-list lines are "<product name> | ..."), without which a
    comparison case's chunks are ambiguous (e.g. two bare "$199.96" /
    "$329.99" price sentences with nothing saying which product either
    one belongs to)."""
    if result is None:
        return []
    if result.exhaustive_matches is not None:
        return [f"{d.product_name}: {d.shown_text}" for m in result.exhaustive_matches for d in m.evidence]
    if result.comparison_results is not None:
        return [f"{sd.document.product_name}: {sd.document.shown_text}" for docs in result.comparison_results.values() for sd in docs]
    return [f"{sd.document.product_name}: {sd.document.shown_text}" for sd in result.results]


def retrieved_chunks(golden: GoldenCase, engine) -> list[str]:
    return chunks_from_retrieval_result(engine.retrieve(golden.question))


@dataclass
class RetrieverCaseResult:
    id: str
    category: str
    exercised: bool
    reason: str
    retrieved_chunk_count: int = 0
    metrics: list[MetricOutcome] = field(default_factory=list)


def evaluate_case(golden: GoldenCase, engine, mode: str) -> RetrieverCaseResult:
    if not has_document_scope(golden):
        return RetrieverCaseResult(
            golden.id, golden.category, False,
            "no document-level gold scope for this category -- not a retrieval question",
        )
    chunks = retrieved_chunks(golden, engine)
    if not chunks:
        return RetrieverCaseResult(golden.id, golden.category, False, "retriever returned nothing for this query")

    model = judge_model()
    test_case = LLMTestCase(input=golden.question, retrieval_context=chunks, expected_output=golden.golden_answer)
    metrics = [
        measure_if_live("ContextualPrecisionMetric", lambda: ContextualPrecisionMetric(model=model), test_case, mode),
        measure_if_live("ContextualRecallMetric", lambda: ContextualRecallMetric(model=model), test_case, mode),
    ]
    if golden.category not in NO_RELEVANT_EVIDENCE_BY_DESIGN_CATEGORIES:
        metrics.append(
            measure_if_live("ContextualRelevancyMetric", lambda: ContextualRelevancyMetric(model=model), test_case, mode)
        )
    return RetrieverCaseResult(golden.id, golden.category, True, "retrieval scored against golden answer",
                                retrieved_chunk_count=len(chunks), metrics=metrics)


def run(goldens: list[GoldenCase], engine, mode: str = "live") -> dict:
    case_results = [evaluate_case(g, engine, mode) for g in goldens]
    all_metrics = [m for c in case_results for m in c.metrics]
    return {
        "judge_model": judge_model(),
        "cases": [
            {
                "id": c.id, "category": c.category, "exercised": c.exercised, "reason": c.reason,
                "retrieved_chunk_count": c.retrieved_chunk_count,
                "metrics": [m.to_dict() for m in c.metrics],
            }
            for c in case_results
        ],
        "total_cases": len(case_results),
        "exercised_cases": sum(1 for c in case_results if c.exercised),
        "aggregate": aggregate_scores(all_metrics),
    }
