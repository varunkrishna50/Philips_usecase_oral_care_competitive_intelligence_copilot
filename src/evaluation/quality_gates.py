"""Configurable quality-gate thresholds, one section per evaluation
category, loaded from data/evaluation/quality_gates.json so thresholds
are editable without a code change. Defaults here are a first cut
(DeepEval's own default per-metric threshold is 0.5; these start at
0.6-0.8 given a GPT-4.1 judge) -- tune after the first real --live run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_GATES_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "quality_gates.json"


@dataclass
class QualityGates:
    retriever: dict
    generator: dict
    pipeline: dict
    application: dict

    @classmethod
    def load(cls, path: Path = DEFAULT_GATES_PATH) -> "QualityGates":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: data.get(k, {}) for k in ("retriever", "generator", "pipeline", "application")})


@dataclass
class GateResult:
    name: str
    threshold: float
    actual: Optional[float]
    passed: bool
    comparison: str  # "min" or "max"


def _mean_score(category_report: dict, metric_name: str) -> Optional[float]:
    agg = (category_report or {}).get("aggregate") or {}
    entry = agg.get(metric_name)
    return entry.get("mean_score") if entry else None


def _pass_rate(category_report: dict, metric_name: str) -> Optional[float]:
    agg = (category_report or {}).get("aggregate") or {}
    entry = agg.get(metric_name)
    return entry.get("pass_rate") if entry else None


def _gate(name: str, threshold, actual, comparison: str) -> GateResult:
    if actual is None:
        return GateResult(name, threshold, None, passed=False, comparison=comparison)
    passed = (actual >= threshold) if comparison == "min" else (actual <= threshold)
    return GateResult(name, threshold, actual, passed, comparison)


def evaluate_gates(gates: QualityGates, report: dict) -> list[GateResult]:
    """`report` is the dict run_deepeval_suite() returns: one key per
    category (retriever/generator/pipeline/application). Gates for a
    category the caller didn't request this invocation (--category
    filtered it out) are skipped entirely -- that's a deliberate
    filter, not a gap. Within a category that DID run, a metric/case
    that genuinely didn't execute still reports actual=None,
    passed=False -- never silently counted as a pass."""
    results: list[GateResult] = []

    retriever, generator, pipeline, application = (
        report.get("retriever"), report.get("generator"), report.get("pipeline"), report.get("application"),
    )

    if retriever is not None:
        for metric, key in (("ContextualPrecisionMetric", "contextual_precision_min"),
                             ("ContextualRecallMetric", "contextual_recall_min"),
                             ("ContextualRelevancyMetric", "contextual_relevancy_min")):
            if key in gates.retriever:
                results.append(_gate(f"retriever.{key}", gates.retriever[key], _mean_score(retriever, metric), "min"))

    if generator is not None:
        for metric, key in (("FaithfulnessMetric", "faithfulness_min"), ("AnswerRelevancyMetric", "answer_relevancy_min"),
                             ("CitationFaithfulness", "citation_faithfulness_min")):
            if key in gates.generator:
                results.append(_gate(f"generator.{key}", gates.generator[key], _mean_score(generator, metric), "min"))

    if pipeline is not None:
        for metric, key in (("FaithfulnessMetric", "faithfulness_min"), ("AnswerRelevancyMetric", "answer_relevancy_min"),
                             ("ContextualRelevancyMetric", "contextual_relevancy_min"), ("Groundedness", "groundedness_min")):
            if key in gates.pipeline:
                results.append(_gate(f"pipeline.{key}", gates.pipeline[key], _mean_score(pipeline, metric), "min"))

    if application is not None:
        for metric, key in (("Correctness", "correctness_min"), ("Completeness", "completeness_min"),
                             ("CitationFaithfulness", "citation_faithfulness_min"),
                             ("Toxicity", "toxicity_min"), ("PIILeakage", "pii_leakage_min"),
                             ("RoleAdherence", "role_adherence_min"), ("NonAdvice", "non_advice_min")):
            if key in gates.application:
                results.append(_gate(f"application.{key}", gates.application[key], _mean_score(application, metric), "min"))
        for metric, key in (("ComparisonArithmetic", "comparison_arithmetic_pass_rate_min"),
                             ("DataLeakage", "data_leakage_pass_rate_min"),
                             ("PromptInjection", "prompt_injection_pass_rate_min")):
            if key in gates.application:
                results.append(_gate(f"application.{key}", gates.application[key], _pass_rate(application, metric), "min"))
        aq = application.get("answer_quality_aggregate") or {}
        for key, actual in (
            ("duplicate_source_rate_max", aq.get("duplicate_source_rate")),
            ("redundant_segment_rate_max", aq.get("redundant_segment_rate")),
            ("jargon_rate_max", aq.get("jargon_rate")),
        ):
            if key in gates.application:
                results.append(_gate(f"application.{key}", gates.application[key], actual, "max"))
        if "template_jargon_max" in gates.application:
            results.append(_gate("application.template_jargon_max", gates.application["template_jargon_max"],
                                  len(application.get("template_jargon") or {}), "max"))

    return results


def all_gates_passed(results: list[GateResult]) -> bool:
    return all(r.passed for r in results)
