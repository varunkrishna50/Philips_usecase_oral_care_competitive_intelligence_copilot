"""3C.9: deterministic comparison evaluation. Gold comparisons are
computed independently from gold_labels.py (parquet-derived), never
from the LLM and never by re-running src.rag.comparison_engine --
the pipeline's own ComparisonResult is then checked AGAINST this
independently-computed gold value, so an arithmetic bug shared by both
sides can't cancel out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .gold_labels import GoldCorpus

NUMERIC_ATTRIBUTES = {"price", "mode_count", "battery_life_days"}
BOOLEAN_ATTRIBUTES = {"pressure_sensing", "app_connectivity", "bluetooth_connectivity", "timer"}


@dataclass
class GoldComparison:
    attribute: str
    status_a: str
    status_b: str
    value_a: Any
    value_b: Any
    comparable: bool
    diff: Optional[float] = None  # numeric attributes only, abs(value_a - value_b)
    a_true: Optional[bool] = None  # boolean attributes only
    b_true: Optional[bool] = None


def compute_gold_comparison(corpus: GoldCorpus, product_a_id: str, product_b_id: str, attribute: str) -> GoldComparison:
    status_a, value_a, _ = corpus.resolve_status(product_a_id, attribute)
    status_b, value_b, _ = corpus.resolve_status(product_b_id, attribute)

    if (attribute in NUMERIC_ATTRIBUTES and status_a == "stated" and status_b == "stated"
            and value_a is not None and value_b is not None):
        return GoldComparison(attribute, status_a, status_b, value_a, value_b, True, diff=abs(value_a - value_b))

    if (attribute in BOOLEAN_ATTRIBUTES and status_a in ("stated", "stated_absent")
            and status_b in ("stated", "stated_absent")):
        a_true = status_a == "stated" and value_a is True
        b_true = status_b == "stated" and value_b is True
        return GoldComparison(attribute, status_a, status_b, value_a, value_b, True, a_true=a_true, b_true=b_true)

    return GoldComparison(attribute, status_a, status_b, value_a, value_b, comparable=False)


@dataclass
class ComparisonCheckResult:
    attribute: str
    passed: bool
    reason: str


def check_pipeline_comparison(gold: GoldComparison, pipeline_result) -> ComparisonCheckResult:
    """pipeline_result is a src.rag.comparison_engine.ComparisonResult."""
    if gold.comparable != pipeline_result.comparable:
        return ComparisonCheckResult(
            gold.attribute, False,
            f"gold comparable={gold.comparable} but pipeline comparable={pipeline_result.comparable}",
        )
    if not gold.comparable:
        return ComparisonCheckResult(gold.attribute, True, "both agree: not safely comparable")

    if gold.attribute in NUMERIC_ATTRIBUTES:
        pipeline_a = pipeline_result.status_a.value
        pipeline_b = pipeline_result.status_b.value
        if pipeline_a is None or pipeline_b is None:
            return ComparisonCheckResult(gold.attribute, False, "pipeline comparable=True but a value is None")
        pipeline_diff = abs(pipeline_a - pipeline_b)
        if abs(pipeline_diff - gold.diff) > 1e-6:
            return ComparisonCheckResult(
                gold.attribute, False,
                f"gold diff={gold.diff} but pipeline diff={pipeline_diff} (a={pipeline_a}, b={pipeline_b})",
            )
        return ComparisonCheckResult(gold.attribute, True, f"diff matches gold ({gold.diff})")

    # boolean
    pipeline_a_true = pipeline_result.status_a.status == "stated" and pipeline_result.status_a.value is True
    pipeline_b_true = pipeline_result.status_b.status == "stated" and pipeline_result.status_b.value is True
    if (pipeline_a_true, pipeline_b_true) != (gold.a_true, gold.b_true):
        return ComparisonCheckResult(
            gold.attribute, False,
            f"gold (a_true={gold.a_true}, b_true={gold.b_true}) but pipeline (a_true={pipeline_a_true}, b_true={pipeline_b_true})",
        )
    return ComparisonCheckResult(gold.attribute, True, "boolean values match gold")
