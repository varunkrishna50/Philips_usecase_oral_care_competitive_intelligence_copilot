"""Oral-B-specific normalization.

Most attribute extraction is brand-agnostic (utils.py). Oral-B-specific
points:

* Evidence comes only from copy a shopper can actually read: the visible
  page text, the FAQ answers, and the product description / "What's in the
  box" / warranty tabs (Step 1 adds those as headed sections because their
  bodies are rendered client-side). Oral-B's internal merchandising fields
  (`TextFeatureN`, `Shop By Feature`, ... in a hidden `dl.productView-info`)
  are NOT product copy and are never used as evidence.
* `included_items` is the "What's in the box" list (see
  detail_attributes.apply_detail_attributes); `specifications` are the
  short bold terms of the description bullets plus the stated
  dimensions/weight.
"""
from __future__ import annotations

from pathlib import Path

from . import utils
from .detail_attributes import apply_detail_attributes
from .evidence import EvidenceStore
from .filters import usable_sections
from .models import NormalizationWarning, NormalizedProduct, VariantOut


def derive_product_id(record: dict) -> str:
    """Reuse Step 1's own filename stem (e.g.
    "oralb_io-series-9-rechargeable-electric-toothbrush") -- this is the
    stable per-MODEL slug Step 1's own variant-grouping already computed,
    not the color-specific JSON-LD SKU (which would change if Step 1 ever
    picked a different representative color on a future run)."""
    return Path(record["raw_html_path"]).stem


def extract_specifications(sections, product_id, record, store, warnings) -> utils.AttributeValue:
    """Short, shopper-visible specification terms: the bold lead-ins of the
    description bullets ("automatic gum pressure sensor", "2-minute brush
    timer", ...) plus the dimensions and weight lines of the box list."""
    details = record.get("details") or {}
    # Bold lead-ins that are really headline claims ("removes 100% more
    # plaque") belong to `claims`, not to the spec list.
    terms = [
        t.rstrip(":").strip()
        for t in (details.get("key_terms") or [])
        if not utils.CLAIM_MARKER.search(t)
    ]
    terms += [t for t in (details.get("dimensions"), details.get("weight")) if t]
    entries = []
    for term in terms:
        for idx, section in sections:
            if term in utils.combined_section_text(section):
                entries.append((term, section, idx))
                break
    return utils.build_specifications(product_id, record, store, entries)


def normalize_product(record: dict, store: EvidenceStore, warnings: list[NormalizationWarning]) -> NormalizedProduct:
    product_id = derive_product_id(record)
    sections = usable_sections(record["sections"], product_id, warnings)

    price_av, currency_av, price_type_av = utils.extract_price(record, product_id, store, warnings)
    pressure_av = utils.extract_pressure_sensing(sections, product_id, record, store, warnings)
    app_av = utils.extract_app_connectivity(sections, product_id, record, store, warnings)
    bluetooth_av = utils.extract_bluetooth_connectivity(sections, product_id, record, store, warnings)
    timer_av, timer_desc_av = utils.extract_timer(sections, product_id, record, store, warnings)
    battery_days_av, battery_raw_av = utils.extract_battery_life(sections, product_id, record, store, warnings)
    charging_av, charging_raw_av = utils.extract_charging_method(sections, product_id, record, store, warnings)
    mode_count_av, modes_av = utils.extract_modes(sections, product_id, record, store, warnings)
    included_items_av = utils.extract_included_items(sections, product_id, record, store, warnings)
    description_av = utils.extract_description(sections, product_id, record, store, warnings)
    claims_av = utils.extract_claims(sections, product_id, record, store, warnings)
    specifications_av = extract_specifications(sections, product_id, record, store, warnings)

    attrs = apply_detail_attributes(
        dict(
            price=price_av, currency=currency_av, price_type=price_type_av,
            mode_count=mode_count_av, modes=modes_av,
            pressure_sensing=pressure_av, app_connectivity=app_av, bluetooth_connectivity=bluetooth_av,
            timer=timer_av, timer_description=timer_desc_av,
            battery_life_days=battery_days_av, battery_life_raw=battery_raw_av,
            charging_method=charging_av, charging_method_raw=charging_raw_av,
            included_items=included_items_av, description=description_av,
            claims=claims_av, specifications=specifications_av,
        ),
        record, product_id, store, warnings,
    )
    return NormalizedProduct(
        product_id=product_id,
        brand=record["brand"],
        product_name=record["product_name"],
        product_url=record["product_url"],
        discovered_url=record["discovered_url"],
        source_listing_url=record["source_listing_url"],
        retrieved_at=record["retrieved_at"],
        raw_html_path=record["raw_html_path"],
        content_hash=record["content_hash"],
        variants=[VariantOut(**v) for v in record.get("variants", [])],
        **attrs,
    )
