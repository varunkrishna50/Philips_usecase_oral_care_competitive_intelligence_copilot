"""Shared plumbing for running DeepEval metrics (and classifiers) across
the 4 eval categories: a uniform MetricOutcome (so every category's
report has the same shape) and a `measure()`/`classify()` that degrade
to not_exercised on any judge-call failure (network error, malformed
judge response, missing OPENAI_API_KEY) instead of crashing the whole
suite -- same philosophy as src/rag/llm_provider.py's single
LLMUnavailable chokepoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from deepeval.classifiers.base_classifier import BaseClassifier
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

from .judge import judge_available


@dataclass
class MetricOutcome:
    name: str
    exercised: bool
    score: Optional[float] = None
    passed: Optional[bool] = None
    reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def measure(metric: BaseMetric, test_case: LLMTestCase, name: Optional[str] = None) -> MetricOutcome:
    metric_name = name or getattr(metric, "name", None) or metric.__class__.__name__
    if not judge_available():
        return MetricOutcome(metric_name, exercised=False, error="OPENAI_API_KEY not set; judge model unavailable")
    try:
        metric.measure(test_case, _show_indicator=False)
    except Exception as exc:  # noqa: BLE001 -- a judge-call failure must degrade to not_exercised, never crash the suite
        return MetricOutcome(metric_name, exercised=False, error=str(exc))
    return MetricOutcome(
        metric_name, exercised=True, score=metric.score,
        passed=metric.is_successful(), reason=getattr(metric, "reason", None),
    )


def measure_if_live(name: str, build_metric, test_case: LLMTestCase, mode: str) -> MetricOutcome:
    """Every DeepEval judge call in this package goes through this
    chokepoint: `mode="offline"` must never place a real network call --
    exactly like src/rag/llm_provider.py's own generation client.

    `build_metric` is a zero-arg callable, not an already-constructed
    metric: DeepEval metric constructors eagerly build a real OpenAI
    client and raise immediately if no key is configured, so
    CONSTRUCTING one must be skipped in offline mode too, not just the
    `.measure()` call -- a bare `judge_available()` check isn't enough
    either, since an OPENAI_API_KEY can be configured in the real
    environment even during a deliberately offline run."""
    if mode != "live":
        return MetricOutcome(name, exercised=False, error="offline mode: DeepEval judge calls are disabled")
    try:
        metric = build_metric()
    except Exception as exc:  # noqa: BLE001 -- construction can fail the same way measure() can
        return MetricOutcome(name, exercised=False, error=str(exc))
    return measure(metric, test_case, name=name)


def classify(classifier: BaseClassifier, test_case: LLMTestCase, expected_label: str, name: Optional[str] = None) -> MetricOutcome:
    metric_name = name or getattr(classifier, "name", None) or classifier.__class__.__name__
    if not judge_available():
        return MetricOutcome(metric_name, exercised=False, error="OPENAI_API_KEY not set; judge model unavailable")
    try:
        classifier.classify(test_case, _show_indicator=False)
    except Exception as exc:  # noqa: BLE001 -- a judge-call failure must degrade to not_exercised, never crash the suite
        return MetricOutcome(metric_name, exercised=False, error=str(exc))
    return MetricOutcome(
        metric_name, exercised=True, score=None,
        passed=classifier.is_successful(expected_label), reason=f"label={classifier.label!r}: {classifier.reason}",
    )


def classify_if_live(name: str, build_classifier, expected_label: str, test_case: LLMTestCase, mode: str) -> MetricOutcome:
    """classify()'s counterpart to measure_if_live() -- same
    construct-nothing-until-mode-is-live guarantee, since classifiers
    eagerly build a real OpenAI client in __init__ exactly like
    metrics do."""
    if mode != "live":
        return MetricOutcome(name, exercised=False, error="offline mode: DeepEval judge calls are disabled")
    try:
        classifier = build_classifier()
    except Exception as exc:  # noqa: BLE001
        return MetricOutcome(name, exercised=False, error=str(exc))
    return classify(classifier, test_case, expected_label, name=name)


def aggregate_scores(outcomes: list[MetricOutcome]) -> dict:
    """{metric_name: {"cases": n, "exercised": n, "mean_score": x, "pass_rate": x}}"""
    by_name: dict[str, list[MetricOutcome]] = {}
    for o in outcomes:
        by_name.setdefault(o.name, []).append(o)
    out = {}
    for name, items in by_name.items():
        exercised = [i for i in items if i.exercised]
        scores = [i.score for i in exercised if i.score is not None]
        passed = [i for i in exercised if i.passed]
        out[name] = {
            "cases": len(items),
            "exercised": len(exercised),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "pass_rate": (len(passed) / len(exercised)) if exercised else None,
        }
    return out
