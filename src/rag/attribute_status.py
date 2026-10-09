"""Per-product-per-attribute status resolution (3A.12) and the
status-aware phrasing templates reused by normal answers (3B.8),
exhaustive answers (3B.9), and the comparison engine (3B.1).

Status comes from the evidence corpus's own `status` field when
evidence exists, and is "not_mentioned" when it doesn't -- never
inferred from whether a retrieval/ranking step happened to surface
something, per 3A.12's core rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import ProductRecord, RetrievalDocument

ATTRIBUTE_LABELS = {
    "price": "price",
    "currency": "currency",
    "price_type": "price type",
    "mode_count": "number of brushing modes",
    "modes": "brushing mode names",
    "pressure_sensing": "pressure sensing",
    "app_connectivity": "app connectivity",
    "bluetooth_connectivity": "Bluetooth connectivity",
    "timer": "timer",
    "timer_description": "timer",
    "battery_life_days": "battery life",
    "battery_life_raw": "battery life",
    "charging_method": "charging method",
    "charging_method_raw": "charging method",
    "included_items": "included items",
    "description": "description",
    "claims": "manufacturer claims",
    "specifications": "specifications",
    "original_price": "original (pre-discount) price",
    "discount_amount": "discount amount",
    "discount_pct": "discount percentage",
    "price_after_discount": "price after discount",
    "promotion_text": "current promotion",
    "color_count": "number of colors",
    "colors": "available colors",
    "rating_avg": "average customer rating",
    "rating_count": "number of customer ratings",
    "mode_details": "brushing mode descriptions",
    "intensity_levels": "intensity levels",
    "warranty_years": "warranty",
    "battery_type": "battery type",
    "charge_time_hours": "charging time",
    "brush_movements_per_min": "brush speed",
    "display_type": "display",
    "replacement_reminder": "brush head replacement reminder",
    "coverage_tracking": "coverage tracking",
    "travel_case_included": "travel case",
    "brush_heads_included": "brush heads included",
    "money_back_guarantee_days": "money-back guarantee",
    "plaque_removal_claim": "plaque removal claim",
    "gum_health_claim": "gum health claim",
    "dimensions": "dimensions",
    "weight": "weight",
    "disclaimers": "disclaimers and footnotes",
    "claims_with_disclaimers": "claims and the disclaimers that qualify them",
    "availability": "stock availability",
    "listed_price_hidden": "price listed in page data but not displayed",
    "pressure_sensing_details": "pressure sensor details",
    "app_connectivity_details": "app and coaching details",
    "bluetooth_details": "Bluetooth connection details",
    "battery_details": "battery details",
    "charging_details": "charging details",
    "coverage_tracking_details": "coverage tracking details",
    "brushing_zones": "number of brushing zones",
    "zone_interval_seconds": "seconds per brushing zone",
    "brush_head_replacement_interval": "brush head replacement interval",
    "cleaning_technology": "motor and cleaning technology",
    "brush_head_compatibility": "brush head compatibility",
    "feel_and_noise": "feel and noise",
    "price_positioning": "price positioning",
    "noise_db": "noise level (dB)",
    "intensity_details": "intensity level descriptions",
    "intensity_count": "number of intensity levels",
    "battery_indicator": "battery indicator",
    "battery_life_note": "battery life (as stated by the manufacturer)",
    "brush_head_design": "brush head design",
}


def attribute_label(attribute: str) -> str:
    return ATTRIBUTE_LABELS.get(attribute, attribute.replace("_", " "))


@dataclass
class AttributeStatus:
    product: ProductRecord
    attribute: str
    status: str  # stated | not_mentioned | stated_absent | ambiguous
    value: Any
    evidence: list[RetrievalDocument] = field(default_factory=list)

    @property
    def product_id(self) -> str:
        return self.product.product_id


def resolve_attribute_status(
    product: ProductRecord,
    attribute: str,
    retrieval_documents: list[RetrievalDocument],
) -> AttributeStatus:
    docs = [d for d in retrieval_documents if d.product_id == product.product_id and d.attribute == attribute]
    if not docs:
        return AttributeStatus(product=product, attribute=attribute, status="not_mentioned", value=None, evidence=[])
    # Step 2 assigns one status per AttributeValue; every evidence record
    # for the same (product, attribute) pair shares it.
    status = docs[0].status
    value = product.attributes.get(attribute)
    return AttributeStatus(product=product, attribute=attribute, status=status, value=value, evidence=docs)


# Asked attribute -> sibling attributes that explain it. Step 2 stores the bare
# flag/label ("Pressure Sensor") under the asked attribute and the substance
# (how it works, numbers, names) under these siblings. Every name must be a
# real column of the comparison table (tests/test_rag_answer_depth.py checks).
RELATED_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "pressure_sensing": ("pressure_sensing_details",),
    "app_connectivity": ("app_connectivity_details",),
    "bluetooth_connectivity": ("bluetooth_details",),
    "timer": ("timer_description", "zone_interval_seconds", "brushing_zones"),
    "timer_description": ("zone_interval_seconds", "brushing_zones"),
    "battery_life_days": ("battery_details", "battery_life_note", "battery_indicator"),
    "battery_life_raw": ("battery_details", "battery_life_note", "battery_indicator"),
    "charging_method": ("charging_details", "charge_time_hours"),
    "charging_method_raw": ("charging_details", "charge_time_hours"),
    "mode_count": ("mode_details",),
    "modes": ("mode_details",),
    "intensity_levels": ("intensity_details",),
    "intensity_count": ("intensity_details",),
    "coverage_tracking": ("coverage_tracking_details",),
    "cleaning_technology": ("brush_head_design",),
    "price": ("original_price", "price_after_discount", "discount_pct", "promotion_text", "availability"),
    # the headline claim is one sentence; the page may state other figures for the same topic
    # (the 4100: "700% more plaque" in the copy, "Up to 5x" in the spec table) and an answer must not pick one
    "plaque_removal_claim": ("claims",),
    "gum_health_claim": ("claims",),
}

# Only the related records about the asked topic: (asked attribute, related attribute) -> pattern
RELATED_TOPIC_FILTERS: dict[tuple[str, str], str] = {
    ("plaque_removal_claim", "claims"): r"plaque",
    ("gum_health_claim", "claims"): r"\bgums?\b|gum health",
}


def expand_attributes(attributes: list[str]) -> list[str]:
    """The asked attributes first, then their related detail attributes (no duplicates)."""
    out: list[str] = []
    for attribute in attributes:
        if attribute not in out:
            out.append(attribute)
    for attribute in attributes:
        for related in RELATED_ATTRIBUTES.get(attribute, ()):
            if related not in out:
                out.append(related)
    return out


def related_evidence(
    product: ProductRecord,
    attribute: str,
    retrieval_documents: list[RetrievalDocument],
    exclude: tuple[str, ...] = (),
) -> list[RetrievalDocument]:
    """Evidence for the sibling attributes of ``attribute`` (never a status statement: a
    missing sibling just means there is nothing extra to say)."""
    wanted = [a for a in RELATED_ATTRIBUTES.get(attribute, ()) if a not in exclude]
    docs: list[RetrievalDocument] = []
    for related in wanted:
        topic = RELATED_TOPIC_FILTERS.get((attribute, related))
        topic_rx = re.compile(topic, re.I) if topic else None
        docs.extend(
            d for d in retrieval_documents
            if d.product_id == product.product_id and d.attribute == related
            and (topic_rx is None or topic_rx.search(d.shown_text))
        )
    return docs


# what a "stated but no figure" sentence says is missing, per attribute (fallback: "a specific figure for <label>")
_UNSTATED_FIGURE = {
    "battery_life_days": "how many days the battery lasts",
    "battery_life_raw": "how many days the battery lasts",
    "charge_time_hours": "how many hours charging takes",
    "charging_method": "how it is charged",
    "charging_method_raw": "how it is charged",
    "mode_count": "how many modes there are",
    "intensity_count": "how many intensity levels there are",
    "warranty_years": "how many years the warranty lasts",
    "brush_movements_per_min": "how many brush movements per minute it makes",
    "price": "a price",
}


def status_aware_statement(attr_status: AttributeStatus) -> str:
    """The phrasing templates for the four statuses, written in code -- never by an LLM -- so they can
    never be collapsed or reworded into something misleading. Plain words: the page, not the pipeline."""
    label = attribute_label(attr_status.attribute)
    name = attr_status.product.product_name

    if attr_status.status == "not_mentioned":
        return f"The page does not mention {label} for {name}."

    if attr_status.status == "stated_absent":
        return f"The page says {name} does not have {label}."

    if attr_status.status == "ambiguous":
        return f"The page gives conflicting information about {label} for {name}, so I can't give one answer."

    # status == "stated"
    if attr_status.value is None:
        quote = " ".join(attr_status.evidence[0].shown_text.split()) if attr_status.evidence else ""
        what = _UNSTATED_FIGURE.get(attr_status.attribute, f"a specific figure for {label}")
        return f"The page does not state {what} for {name}. It says: \u201c{quote}\u201d"

    if isinstance(attr_status.value, bool):
        return f"{name} has {label}." if attr_status.value else f"{name} does not have {label}."
    return f"{name}: {label} is {_format_value(attr_status.attribute, attr_status.value)}."


def _format_value(attribute: str, value: Any) -> str:
    if isinstance(value, bool):
        return "this feature" if value else "no such feature"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if attribute == "price":
        return f"${value:.2f}"
    return str(value)


def evidence_ids_for(attr_status: AttributeStatus) -> list[str]:
    return [d.evidence_id for d in attr_status.evidence]
