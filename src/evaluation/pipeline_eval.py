"""Category 3 -- Full RAG pipeline: runs the full question flow exactly
as production does (contextualization/query-understanding -> hybrid
retrieval -> rerank+filter -> generation, all inside the real
`answer_query()`) and scores how retrieval and generation work
TOGETHER: the RAG triad (Faithfulness, Answer Relevancy, Contextual
Relevancy) plus a GEval "Groundedness" custom metric.

Must-abstain cases (out_of_corpus, medical_advice, prompt-injection) are
included in this category's case set, not filtered out -- correctly
abstaining is part of what "the pipeline works end to end" means here.
(Retriever-alone's Precision/Recall-against-gold and Application's
Correctness/Completeness-against-gold are deliberately NOT run here;
this category measures the pipeline's internal consistency -- is the
answer faithful to and relevant given whatever it retrieved -- not
correctness against an external gold reference.)

Groundedness is a GEval metric (not the old regex-based checker it
replaces): context=golden_context (this case's hand-written
expected_behavior + must_include from test_cases.json), not whatever
was actually retrieved -- so it catches a claim that's "faithful" to a
wrongly-retrieved context but still false against the case's own
stated ground truth, which FaithfulnessMetric alone cannot.

Takes the one real `answer_query()` AnswerResult per case that
runner.py already computed and shares across this module and
application_eval.py -- never re-runs generation per category.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from .deepeval_support import MetricOutcome, aggregate_scores, measure_if_live
from .golden_dataset import GoldenCase
from .judge import judge_model
from .retriever_eval import chunks_from_retrieval_result

GROUNDEDNESS_CRITERIA = (
    "The 'context' is independently-sourced, known-true facts about the product(s) named in the "
    "'input' question. Determine whether the 'actual output' contains any product name or numeric "
    "claim (price, mode count, battery life, or similar figure) that is NOT supported by, or "
    "contradicts, the given context. An answer with no unsupported or contradicting claim is fully "
    "grounded, regardless of whether it also cites other sources."
)


@dataclass
class PipelineCaseResult:
    id: str
    category: str
    exercised: bool
    reason: str
    retrieval_context_size: int = 0
    metrics: list[MetricOutcome] = field(default_factory=list)


def evaluate_case(golden: GoldenCase, answer, mode: str) -> PipelineCaseResult:
    actual_output = (answer.answer_text or "").strip()
    if not actual_output:
        return PipelineCaseResult(golden.id, golden.category, False, "pipeline produced no answer text")

    retrieval_context = chunks_from_retrieval_result(answer.retrieval_result)
    model = judge_model()
    rag_test_case = LLMTestCase(input=golden.question, actual_output=actual_output, retrieval_context=retrieval_context)

    metrics = [measure_if_live("AnswerRelevancyMetric", lambda: AnswerRelevancyMetric(model=model), rag_test_case, mode)]
    if retrieval_context:
        metrics.append(measure_if_live("FaithfulnessMetric", lambda: FaithfulnessMetric(model=model), rag_test_case, mode))
        metrics.append(measure_if_live("ContextualRelevancyMetric", lambda: ContextualRelevancyMetric(model=model), rag_test_case, mode))
    else:
        metrics.append(MetricOutcome("FaithfulnessMetric", exercised=False, error="no retrieval_context (abstained before/without retrieving anything)"))
        metrics.append(MetricOutcome("ContextualRelevancyMetric", exercised=False, error="no retrieval_context (abstained before/without retrieving anything)"))

    if golden.golden_context:
        groundedness_case = LLMTestCase(input=golden.question, actual_output=actual_output, context=golden.golden_context)
        metrics.append(measure_if_live(
            "Groundedness",
            lambda: GEval(name="Groundedness", criteria=GROUNDEDNESS_CRITERIA,
                          evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.CONTEXT],
                          model=model),
            groundedness_case, mode,
        ))
    else:
        metrics.append(MetricOutcome("Groundedness", exercised=False, error="no independently-loaded gold context for this category"))

    return PipelineCaseResult(
        golden.id, golden.category, True, "full pipeline scored end to end",
        retrieval_context_size=len(retrieval_context), metrics=metrics,
    )


def run(goldens: list[GoldenCase], answers: dict[str, object], mode: str = "live") -> dict:
    case_results = [evaluate_case(g, answers[g.id], mode) for g in goldens]
    all_metrics = [m for c in case_results for m in c.metrics]
    return {
        "judge_model": judge_model(),
        "cases": [
            {
                "id": c.id, "category": c.category, "exercised": c.exercised, "reason": c.reason,
                "retrieval_context_size": c.retrieval_context_size,
                "metrics": [m.to_dict() for m in c.metrics],
            }
            for c in case_results
        ],
        "total_cases": len(case_results),
        "exercised_cases": sum(1 for c in case_results if c.exercised),
        "aggregate": aggregate_scores(all_metrics),
    }
