"""3C.9: deterministic comparison evaluation. Gold comparisons are read
straight from the real, already-loaded product catalog (ProductRecord's
own `attributes` dict) -- the pipeline's own ComparisonResult is then
checked against these values, so a wrong-sign or off-by-one bug in
compare_attribute's diff/boolean logic still gets caught.

This reads a null attribute value as simply "not comparable", with no
separate not_mentioned/stated_absent/ambiguous distinction -- that
distinction needed the independent evidence.json reader this module no
longer uses (dropped at the user's request, see golden_dataset.py). It
is not a real loss here: of the four Step 2 statuses, "stated_absent"
and "ambiguous" have zero real occurrences in this corpus (confirmed by
direct inspection when the independent reader still existed), so a null
value in the catalog has only ever actually meant not_mentioned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.rag.models import ProductRecord

NUMERIC_ATTRIBUTES = {"price", "mode_count", "battery_life_days"}
BOOLEAN_ATTRIBUTES = {"pressure_sensing", "app_connectivity", "bluetooth_connectivity", "timer"}


@dataclass
class GoldComparison:
    attribute: str
    value_a: Any
    value_b: Any
    comparable: bool
    diff: Optional[float] = None  # numeric attributes only, abs(value_a - value_b)
    a_true: Optional[bool] = None  # boolean attributes only
    b_true: Optional[bool] = None


def _find_product(product_corpus: list[ProductRecord], product_id: str) -> Optional[ProductRecord]:
    return next((p for p in product_corpus if p.product_id == product_id), None)


def compute_gold_comparison(
    product_corpus: list[ProductRecord], product_a_id: str, product_b_id: str, attribute: str,
) -> GoldComparison:
    product_a = _find_product(product_corpus, product_a_id)
    product_b = _find_product(product_corpus, product_b_id)
    value_a = product_a.get(attribute) if product_a else None
    value_b = product_b.get(attribute) if product_b else None

    if attribute in NUMERIC_ATTRIBUTES and value_a is not None and value_b is not None:
        return GoldComparison(attribute, value_a, value_b, True, diff=abs(value_a - value_b))

    if attribute in BOOLEAN_ATTRIBUTES and value_a is not None and value_b is not None:
        return GoldComparison(attribute, value_a, value_b, True, a_true=bool(value_a), b_true=bool(value_b))

    return GoldComparison(attribute, value_a, value_b, comparable=False)


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
