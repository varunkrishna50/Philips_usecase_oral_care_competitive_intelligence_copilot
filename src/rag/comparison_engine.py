"""Deterministic comparison engine (3B.1).

All arithmetic (differences, which product has more/fewer) happens HERE
in plain Python, never inside an LLM prompt -- the LLM (3b, when
available) only ever verbalizes a comparison this module already
computed; the no-key fallback (3B.3) uses this module's `summary`
directly as the answer text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .attribute_status import AttributeStatus, attribute_label, evidence_ids_for, resolve_attribute_status, status_aware_statement
from .models import ProductRecord, RetrievalDocument

NUMERIC_ATTRIBUTES = {"price", "mode_count", "battery_life_days"}
BOOLEAN_ATTRIBUTES = {"pressure_sensing", "app_connectivity", "bluetooth_connectivity", "timer"}


@dataclass
class ComparisonResult:
    attribute: str
    status_a: AttributeStatus
    status_b: AttributeStatus
    comparable: bool
    summary: str
    citation_evidence_ids: list[str] = field(default_factory=list)


def _format_number(attribute: str, value) -> str:
    return f"${value:.2f}" if attribute == "price" else str(value)


def compare_attribute(
    product_a: ProductRecord,
    product_b: ProductRecord,
    attribute: str,
    retrieval_documents: list[RetrievalDocument],
) -> ComparisonResult:
    status_a = resolve_attribute_status(product_a, attribute, retrieval_documents)
    status_b = resolve_attribute_status(product_b, attribute, retrieval_documents)
    label = attribute_label(attribute)

    both_stated_numeric = (
        attribute in NUMERIC_ATTRIBUTES
        and status_a.status == "stated" and status_b.status == "stated"
        and status_a.value is not None and status_b.value is not None
    )
    if both_stated_numeric:
        diff = status_a.value - status_b.value
        if diff == 0:
            summary = (
                f"{product_a.product_name} and {product_b.product_name} both have the same stated "
                f"{label}: {_format_number(attribute, status_a.value)}."
            )
        else:
            higher, lower, higher_val, lower_val = (
                (product_a, product_b, status_a.value, status_b.value) if diff > 0
                else (product_b, product_a, status_b.value, status_a.value)
            )
            gap = higher_val - lower_val
            summary = (
                f"{higher.product_name} has a stated {label} of {_format_number(attribute, higher_val)}, "
                f"while {lower.product_name} has {_format_number(attribute, lower_val)} -- a difference of "
                f"{_format_number(attribute, gap)}."
            )
        return ComparisonResult(
            attribute=attribute, status_a=status_a, status_b=status_b, comparable=True, summary=summary,
            citation_evidence_ids=evidence_ids_for(status_a) + evidence_ids_for(status_b),
        )

    both_stated_boolean = (
        attribute in BOOLEAN_ATTRIBUTES
        and status_a.status in ("stated", "stated_absent")
        and status_b.status in ("stated", "stated_absent")
    )
    if both_stated_boolean:
        a_true = status_a.status == "stated" and status_a.value is True
        b_true = status_b.status == "stated" and status_b.value is True
        if a_true and b_true:
            summary = f"Both {product_a.product_name} and {product_b.product_name} have {label}."
        elif not a_true and not b_true:
            summary = f"Neither {product_a.product_name} nor {product_b.product_name} has {label}."
        elif a_true:
            summary = f"{product_a.product_name} has {label}, while {product_b.product_name} does not."
        else:
            summary = f"{product_b.product_name} has {label}, while {product_a.product_name} does not."
        return ComparisonResult(
            attribute=attribute, status_a=status_a, status_b=status_b, comparable=True, summary=summary,
            citation_evidence_ids=evidence_ids_for(status_a) + evidence_ids_for(status_b),
        )

    # Not safely comparable: at least one side is not_mentioned/ambiguous/
    # stated-with-null, or a genuine conflict -- never guess or average.
    parts = [status_aware_statement(status_a), status_aware_statement(status_b)]
    summary = " ".join(parts) + " A side-by-side comparison isn't possible for this one."
    return ComparisonResult(
        attribute=attribute, status_a=status_a, status_b=status_b, comparable=False, summary=summary,
        citation_evidence_ids=evidence_ids_for(status_a) + evidence_ids_for(status_b),
    )
