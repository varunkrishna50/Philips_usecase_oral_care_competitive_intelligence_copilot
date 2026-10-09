"""Per-product-per-attribute status resolution (3A.12) and the
status-aware phrasing templates reused by normal answers (3B.8),
exhaustive answers (3B.9), and the comparison engine (3B.1).

Status comes from the evidence corpus's own `status` field when
evidence exists, and is "not_mentioned" when it doesn't -- never
inferred from whether a retrieval/ranking step happened to surface
something, per 3A.12's core rule.
"""
from __future__ import annotations

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


def status_aware_statement(attr_status: AttributeStatus) -> str:
    """The exact phrasing templates from 3A.12/3B.8, verbalized in code
    -- never by an LLM -- so the four statuses can never be collapsed or
    reworded into something misleading."""
    label = attribute_label(attr_status.attribute)
    name = attr_status.product.product_name

    if attr_status.status == "not_mentioned":
        return f"The available source does not mention {label} for {name}."

    if attr_status.status == "stated_absent":
        return f"The source explicitly indicates that {label} is not present for {name}."

    if attr_status.status == "ambiguous":
        return (
            f"The available sources contain conflicting or ambiguous information about {label} "
            f"for {name}, so I cannot establish a single value."
        )

    # status == "stated"
    if attr_status.value is None:
        quote = attr_status.evidence[0].evidence_text if attr_status.evidence else ""
        return (
            f'The manufacturer states: "{quote}" regarding {label} for {name}, but the source does not '
            f"provide a normalized value."
        )

    return f"{name} has {_format_value(attr_status.attribute, attr_status.value)} for {label}."


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
