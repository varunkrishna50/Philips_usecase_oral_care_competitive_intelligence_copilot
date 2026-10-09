"""Philips-specific normalization.

Most attribute extraction is brand-agnostic (utils.py) since both brands
use the same vocabulary ("pressure sensor", "magnetic charger", ...) in
their own product-specific sections. The one Philips-specific piece is
"specifications": a short feature-highlight bullet row that sits right
after the H1 on most pages, duplicated line-by-line (a responsive
mobile/desktop DOM artifact from Step 1's extraction) and needs
de-duplication before being kept as a flat spec list.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import utils
from .detail_attributes import apply_detail_attributes
from .evidence import EvidenceStore
from .filters import usable_sections
from .models import NormalizationWarning, NormalizedProduct, VariantOut

SHORT_BULLET_RE = re.compile(r"^.{3,60}$")


def derive_product_id(record: dict) -> str:
    """Reuse Step 1's own filename stem (e.g. "philips_hx9990-13") -- it's
    already proven deterministic (Step 1's cache depends on it being
    stable across runs) and already derived from the product's SKU."""
    return Path(record["raw_html_path"]).stem


def _dedupe_consecutive(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if out and out[-1] == line:
            continue
        out.append(line)
    return out


def extract_specifications(sections, product_id, record, store, warnings) -> utils.AttributeValue:
    entries = []
    for idx, section in sections:
        lines = _dedupe_consecutive([line.strip() for line in section["text"].split("\n")])
        for line in lines:
            if not SHORT_BULLET_RE.match(line):
                continue
            if utils._is_noise_line(line):
                continue
            if line.endswith((".", "*")):
                continue
            if utils.CLAIM_MARKER.search(line):
                continue
            entries.append((line, section, idx))
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
