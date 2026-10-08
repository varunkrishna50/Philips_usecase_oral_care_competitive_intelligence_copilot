"""Oral-B-specific normalization.

Most attribute extraction is brand-agnostic (utils.py). The Oral-B-
specific piece is reading two structured `<label>:\n<value>` attribute
lists rendered into the page's visible text: `TextFeatureN:` (present on
9 of 13 products) and `Shop By Feature:` (a second, overlapping list with
some items the first doesn't have, e.g. "1 Brush", "1 refill head", "1
Magnetic Travel Pouch"). Both are curated, product-specific attribute
pairs (not free-form prose), so each value is classified as:

1. an explicit "<N> <item>" package-contents line -> included_items
   (utils.extract_included_items already covers this from the same
   usable sections generically, for either brand);
2. a bare accessory name with no leading digit (e.g. "Charging Station",
   "iO Senser(tm) Smart Charger") -> also included_items, via
   extract_labeled_included_items() below. The no-digit relaxation is
   deliberately scoped ONLY to these curated label:value pairs, never to
   free prose, where "comes with X" phrasing is far more ambiguous (see
   README for the "What's in the box" investigation -- that section
   turned out to be marketing copy, not an itemized list, hence this
   narrower, data-driven source instead);
3. everything else -> specifications.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import utils
from .evidence import EvidenceStore
from .filters import usable_sections
from .models import AttributeValue, NormalizationWarning, NormalizedProduct, VariantOut

LABELED_FEATURE_LABEL = re.compile(r"^(?:TextFeature\d+|Shop By Feature):$")
# A bare accessory name, with no leading digit, that's still an explicit
# physical component (as opposed to a software/capability feature like
# "Bluetooth Connectivity" or "7 Smart Brushing Modes").
BARE_ACCESSORY_ITEM = re.compile(
    r"charger$|charging (?:station|case|travel case)$|travel case$|travel pouch$", re.I,
)


def derive_product_id(record: dict) -> str:
    """Reuse Step 1's own filename stem (e.g.
    "oralb_io-series-9-rechargeable-electric-toothbrush") -- this is the
    stable per-MODEL slug Step 1's own variant-grouping already computed,
    not the color-specific JSON-LD SKU (which would change if Step 1 ever
    picked a different representative color on a future run)."""
    return Path(record["raw_html_path"]).stem


def _labeled_feature_values(sections) -> list[tuple[str, dict, int]]:
    """(value_line, section, index) for every TextFeatureN:/"Shop By
    Feature:" labeled pair in usable sections, in page order."""
    results = []
    for idx, section in sections:
        lines = section["text"].split("\n")
        for i, line in enumerate(lines):
            if not LABELED_FEATURE_LABEL.match(line.strip()):
                continue
            if i + 1 >= len(lines):
                continue
            value_line = lines[i + 1].strip()
            if value_line:
                results.append((value_line, section, idx))
    return results


def extract_specifications(sections, product_id, record, store, warnings) -> utils.AttributeValue:
    entries = []
    for value_line, section, idx in _labeled_feature_values(sections):
        if utils.INCLUDED_ITEM_LINE.match(value_line):
            continue  # handled by extract_included_items instead
        if BARE_ACCESSORY_ITEM.search(value_line):
            continue  # handled by extract_labeled_included_items instead
        entries.append((value_line, section, idx))
    return utils.build_specifications(product_id, record, store, entries)


def extract_labeled_included_items(sections, product_id, record, store, warnings) -> tuple[list[str], list[str]]:
    """Bare accessory names (no leading digit) from the labeled
    TextFeatureN/"Shop By Feature" pairs. Returns (values, evidence_ids)
    so the caller can merge these with utils.extract_included_items()'s
    digit-prefixed findings into one AttributeValue."""
    items: list[str] = []
    eids: list[str] = []
    seen: set[str] = set()
    for value_line, section, idx in _labeled_feature_values(sections):
        if not BARE_ACCESSORY_ITEM.search(value_line):
            continue
        if value_line in seen:
            continue
        seen.add(value_line)
        combined = utils.combined_section_text(section)
        if value_line not in combined:
            continue
        eid = store.add_from_section(
            product_id=product_id, attribute="included_items", value=value_line, status="stated",
            raw_value=value_line, evidence_text=value_line,
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=section["heading"], section_index=idx,
            section_combined_text=combined,
        )
        items.append(value_line)
        eids.append(eid)
    return items, eids


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
    labeled_items, labeled_eids = extract_labeled_included_items(sections, product_id, record, store, warnings)
    if labeled_items:
        existing_values = list(included_items_av.value or [])
        new_pairs = [
            (v, e) for v, e in zip(labeled_items, labeled_eids)
            if not any(v.lower() in existing.lower() for existing in existing_values)
        ]
        if new_pairs:
            merged_values = existing_values + [v for v, _ in new_pairs]
            merged_eids = included_items_av.evidence_ids + [e for _, e in new_pairs]
            included_items_av = AttributeValue(value=merged_values, status="stated", raw_value=None, evidence_ids=merged_eids)
    description_av = utils.extract_description(sections, product_id, record, store, warnings)
    claims_av = utils.extract_claims(sections, product_id, record, store, warnings)
    specifications_av = extract_specifications(sections, product_id, record, store, warnings)

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
        price=price_av,
        currency=currency_av,
        price_type=price_type_av,
        mode_count=mode_count_av,
        modes=modes_av,
        pressure_sensing=pressure_av,
        app_connectivity=app_av,
        bluetooth_connectivity=bluetooth_av,
        timer=timer_av,
        timer_description=timer_desc_av,
        battery_life_days=battery_days_av,
        battery_life_raw=battery_raw_av,
        charging_method=charging_av,
        charging_method_raw=charging_raw_av,
        included_items=included_items_av,
        description=description_av,
        claims=claims_av,
        specifications=specifications_av,
        variants=[VariantOut(**v) for v in record.get("variants", [])],
    )
