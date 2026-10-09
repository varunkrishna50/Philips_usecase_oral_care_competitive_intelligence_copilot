"""Pydantic schema for Step 2: normalized products, evidence, and warnings.

Every attribute that can be derived from source content is represented as
an AttributeValue carrying both a value and a status (stated /
not_mentioned / stated_absent / ambiguous), plus the ids of the Evidence
records that support it. status="stated" does NOT require value to be
non-null -- a topic can be explicitly discussed without yielding a value
that can be safely normalized (see evidence.py / utils.py).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Status = Literal["stated", "not_mentioned", "stated_absent", "ambiguous"]

NOT_MENTIONED = "not_mentioned"


class AttributeValue(BaseModel):
    value: Any = None
    status: Status = NOT_MENTIONED
    raw_value: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)


def not_mentioned() -> AttributeValue:
    return AttributeValue(value=None, status=NOT_MENTIONED, raw_value=None, evidence_ids=[])


class EvidenceRef(BaseModel):
    id: str
    product_id: str
    attribute: str
    value: Any
    status: Status
    raw_value: Optional[str] = None
    evidence_text: str
    source_url: str
    source_file: str
    source_section: Optional[str] = None
    source_section_index: Optional[int] = None
    # Traceability (filled by traceability.annotate_evidence after the build):
    # where the quote lives in a saved text snapshot, and how citable it is.
    snapshot_file: Optional[str] = None
    snapshot_kind: Optional[str] = None  # "sections" | "structured"
    source_offset: Optional[int] = None
    region: Optional[str] = None  # e.g. "Product description", "spec:Cleaning performance", "json_ld"
    visibility: Optional[str] = None  # visible | expanded | hidden
    evidence_tier: Optional[str] = None  # cited | cross_check_only


class VariantOut(BaseModel):
    url: str
    label: Optional[str] = None
    sku: Optional[str] = None


class NormalizationWarning(BaseModel):
    product_id: str
    code: str
    message: str
    attribute: Optional[str] = None
    section_heading: Optional[str] = None


class NormalizedProduct(BaseModel):
    product_id: str
    brand: str
    product_name: str
    product_url: str
    discovered_url: str
    source_listing_url: str
    retrieved_at: str
    raw_html_path: str
    content_hash: str

    price: AttributeValue = Field(default_factory=not_mentioned)
    currency: AttributeValue = Field(default_factory=not_mentioned)
    price_type: AttributeValue = Field(default_factory=not_mentioned)

    mode_count: AttributeValue = Field(default_factory=not_mentioned)
    modes: AttributeValue = Field(default_factory=not_mentioned)

    pressure_sensing: AttributeValue = Field(default_factory=not_mentioned)

    app_connectivity: AttributeValue = Field(default_factory=not_mentioned)
    bluetooth_connectivity: AttributeValue = Field(default_factory=not_mentioned)

    timer: AttributeValue = Field(default_factory=not_mentioned)
    timer_description: AttributeValue = Field(default_factory=not_mentioned)

    battery_life_days: AttributeValue = Field(default_factory=not_mentioned)
    battery_life_raw: AttributeValue = Field(default_factory=not_mentioned)

    charging_method: AttributeValue = Field(default_factory=not_mentioned)
    charging_method_raw: AttributeValue = Field(default_factory=not_mentioned)

    included_items: AttributeValue = Field(default_factory=not_mentioned)

    description: AttributeValue = Field(default_factory=not_mentioned)
    claims: AttributeValue = Field(default_factory=not_mentioned)
    specifications: AttributeValue = Field(default_factory=not_mentioned)

    # Detail attributes (see detail_attributes.py): pricing detail, colors,
    # ratings, mode descriptions, warranty, battery/charging/speed specs,
    # extra smart features, included hardware, headline claims.
    original_price: AttributeValue = Field(default_factory=not_mentioned)
    discount_amount: AttributeValue = Field(default_factory=not_mentioned)
    discount_pct: AttributeValue = Field(default_factory=not_mentioned)
    price_after_discount: AttributeValue = Field(default_factory=not_mentioned)
    promotion_text: AttributeValue = Field(default_factory=not_mentioned)
    color_count: AttributeValue = Field(default_factory=not_mentioned)
    colors: AttributeValue = Field(default_factory=not_mentioned)
    rating_avg: AttributeValue = Field(default_factory=not_mentioned)
    rating_count: AttributeValue = Field(default_factory=not_mentioned)
    mode_details: AttributeValue = Field(default_factory=not_mentioned)
    intensity_levels: AttributeValue = Field(default_factory=not_mentioned)
    warranty_years: AttributeValue = Field(default_factory=not_mentioned)
    battery_type: AttributeValue = Field(default_factory=not_mentioned)
    charge_time_hours: AttributeValue = Field(default_factory=not_mentioned)
    brush_movements_per_min: AttributeValue = Field(default_factory=not_mentioned)
    display_type: AttributeValue = Field(default_factory=not_mentioned)
    replacement_reminder: AttributeValue = Field(default_factory=not_mentioned)
    coverage_tracking: AttributeValue = Field(default_factory=not_mentioned)
    travel_case_included: AttributeValue = Field(default_factory=not_mentioned)
    brush_heads_included: AttributeValue = Field(default_factory=not_mentioned)
    money_back_guarantee_days: AttributeValue = Field(default_factory=not_mentioned)
    plaque_removal_claim: AttributeValue = Field(default_factory=not_mentioned)
    gum_health_claim: AttributeValue = Field(default_factory=not_mentioned)
    brush_head_design: AttributeValue = Field(default_factory=not_mentioned)
    battery_life_note: AttributeValue = Field(default_factory=not_mentioned)
    dimensions: AttributeValue = Field(default_factory=not_mentioned)
    weight: AttributeValue = Field(default_factory=not_mentioned)
    disclaimers: AttributeValue = Field(default_factory=not_mentioned)
    claims_with_disclaimers: AttributeValue = Field(default_factory=not_mentioned)
    availability: AttributeValue = Field(default_factory=not_mentioned)
    listed_price_hidden: AttributeValue = Field(default_factory=not_mentioned)
    pressure_sensing_details: AttributeValue = Field(default_factory=not_mentioned)
    app_connectivity_details: AttributeValue = Field(default_factory=not_mentioned)
    bluetooth_details: AttributeValue = Field(default_factory=not_mentioned)
    battery_details: AttributeValue = Field(default_factory=not_mentioned)
    charging_details: AttributeValue = Field(default_factory=not_mentioned)
    coverage_tracking_details: AttributeValue = Field(default_factory=not_mentioned)
    brushing_zones: AttributeValue = Field(default_factory=not_mentioned)
    zone_interval_seconds: AttributeValue = Field(default_factory=not_mentioned)
    brush_head_replacement_interval: AttributeValue = Field(default_factory=not_mentioned)
    cleaning_technology: AttributeValue = Field(default_factory=not_mentioned)
    brush_head_compatibility: AttributeValue = Field(default_factory=not_mentioned)
    feel_and_noise: AttributeValue = Field(default_factory=not_mentioned)
    price_positioning: AttributeValue = Field(default_factory=not_mentioned)
    noise_db: AttributeValue = Field(default_factory=not_mentioned)
    intensity_details: AttributeValue = Field(default_factory=not_mentioned)
    intensity_count: AttributeValue = Field(default_factory=not_mentioned)
    battery_indicator: AttributeValue = Field(default_factory=not_mentioned)

    variants: list[VariantOut] = Field(default_factory=list)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flattens AttributeValue fields to their bare .value (plus the
        two *_raw passthroughs already modeled as separate fields) for
        CSV/Parquet. Status and evidence_ids are deliberately NOT
        included here -- they live in evidence.json."""
        flat: dict[str, Any] = {
            "product_id": self.product_id,
            "brand": self.brand,
            "product_name": self.product_name,
            "product_url": self.product_url,
            "discovered_url": self.discovered_url,
            "source_listing_url": self.source_listing_url,
            "retrieved_at": self.retrieved_at,
        }
        for field_name in ATTRIBUTE_FIELDS:
            flat[field_name] = getattr(self, field_name).value
        # price_raw has no dedicated AttributeValue of its own (price's
        # raw_value already carries it) but the required schema lists it
        # as its own column, alongside the battery/charging *_raw pairs.
        flat["price_raw"] = self.price.raw_value
        flat["variants"] = [v.model_dump() for v in self.variants]
        return flat


# Names of the AttributeValue-typed fields, used by validation/CSV/Parquet
# writers so they don't have to hard-code the list twice.
ATTRIBUTE_FIELDS = [
    "price",
    "currency",
    "price_type",
    "mode_count",
    "modes",
    "pressure_sensing",
    "app_connectivity",
    "bluetooth_connectivity",
    "timer",
    "timer_description",
    "battery_life_days",
    "battery_life_raw",
    "charging_method",
    "charging_method_raw",
    "included_items",
    "description",
    "claims",
    "specifications",
    "original_price",
    "discount_amount",
    "discount_pct",
    "price_after_discount",
    "promotion_text",
    "color_count",
    "colors",
    "rating_avg",
    "rating_count",
    "mode_details",
    "intensity_levels",
    "warranty_years",
    "battery_type",
    "charge_time_hours",
    "brush_movements_per_min",
    "display_type",
    "replacement_reminder",
    "coverage_tracking",
    "travel_case_included",
    "brush_heads_included",
    "money_back_guarantee_days",
    "plaque_removal_claim",
    "gum_health_claim",
    "brush_head_design",
    "battery_life_note",
    "dimensions",
    "weight",
    "disclaimers",
    "claims_with_disclaimers",
    "availability",
    "listed_price_hidden",
    "pressure_sensing_details",
    "app_connectivity_details",
    "bluetooth_details",
    "battery_details",
    "charging_details",
    "coverage_tracking_details",
    "brushing_zones",
    "zone_interval_seconds",
    "brush_head_replacement_interval",
    "cleaning_technology",
    "brush_head_compatibility",
    "feel_and_noise",
    "price_positioning",
    "noise_db",
    "intensity_details",
    "intensity_count",
    "battery_indicator",
]
