#!/usr/bin/env python3
"""Step 2: normalize Step 1's raw acquisition data into a grounded,
auditable comparison dataset.

Reads data/raw/ only -- never re-scrapes the websites. Writes (data/processed/):

    comparison_table.csv          review view: wide table, tri-state flags, audit columns
    comparison_table.parquet      same attributes WITHOUT the CSV-only columns (RAG input)
    evidence.json                 every verbatim quote + snapshot location + tier
    normalization_warnings.json
    claims_long.csv  specs_long.csv  feature_headlines.csv  included_items_long.csv
    product_aliases.csv  column_dictionary.csv  column_coverage.csv
    snapshots/<id>.structured.txt text blobs for non-section evidence offsets

and runs the quality gate (src/normalization/quality_gate.py) over the files it
just wrote; errors make the build exit non-zero.

Usage:
    python scripts/build_table.py [--verbose]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from normalization.aliases import alias_groups, build_aliases  # noqa: E402
from normalization.longtables import (  # noqa: E402
    build_claims_long,
    build_feature_headlines,
    build_included_items_long,
    build_specs_long,
)
from normalization.models import ATTRIBUTE_FIELDS, AttributeValue, NormalizedProduct  # noqa: E402
from normalization.normalize import index_raw_records, load_raw_records, normalize_all  # noqa: E402
from normalization.quality_gate import FINGERPRINT_COLUMNS, FLAG_COLUMNS, check_files  # noqa: E402
from normalization.traceability import annotate_evidence, annotate_headlines  # noqa: E402
from normalization.validation import validate  # noqa: E402
from normalization.vocab import canonicalize_modes  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Attribute columns (parquet AND csv). Evidence-backed.
CSV_COLUMNS = [
    "product_id", "brand", "product_name", "product_url", "discovered_url", "source_listing_url", "retrieved_at",
    "price", "price_raw", "currency", "price_type", "mode_count", "modes", "pressure_sensing", "app_connectivity",
    "bluetooth_connectivity", "timer", "timer_description", "battery_life_days", "battery_life_raw", "charging_method",
    "charging_method_raw", "included_items", "description", "claims", "specifications", "original_price",
    "discount_amount", "discount_pct", "price_after_discount", "promotion_text", "color_count", "colors", "rating_avg",
    "rating_count", "mode_details", "intensity_levels", "warranty_years", "battery_type", "charge_time_hours",
    "brush_movements_per_min", "display_type", "replacement_reminder", "coverage_tracking", "travel_case_included",
    "brush_heads_included", "money_back_guarantee_days", "plaque_removal_claim", "gum_health_claim", "dimensions",
    "weight", "disclaimers", "claims_with_disclaimers", "availability", "listed_price_hidden",
    "pressure_sensing_details", "app_connectivity_details", "bluetooth_details", "battery_details", "charging_details",
    "coverage_tracking_details", "brushing_zones", "zone_interval_seconds", "brush_head_replacement_interval",
    "cleaning_technology", "brush_head_compatibility", "feel_and_noise", "price_positioning", "noise_db",
    "intensity_details", "intensity_count", "battery_indicator", "brush_head_design", "battery_life_note", "variants",
]

# Audit / derived columns: written to the CSV only, never to the parquet.
CSV_ONLY_COLUMNS = [
    "product_type", "handle_count", "alias_group", "canonical_product_id",
    "list_price", "current_price", "price_as_of", "price_source", "availability_raw",
    "source_snapshot", "text_snapshot", "content_hash",
    "source_variant_url", "source_variant_sku", "source_variant_label", "variant_values",
    "merchandising_tags", "modes_canonical",
]

JSON_ENCODED_COLUMNS = {
    "modes", "included_items", "claims", "specifications", "variants", "colors", "mode_details", "intensity_levels",
    "disclaimers", "claims_with_disclaimers", "pressure_sensing_details", "app_connectivity_details",
    "bluetooth_details", "battery_details", "charging_details", "coverage_tracking_details", "intensity_details",
}
CSV_JSON_ONLY = {"merchandising_tags", "variant_values", "modes_canonical"}

logger = logging.getLogger("normalization")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _csv_cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column in JSON_ENCODED_COLUMNS or column in CSV_JSON_ONLY:
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def flag_cell(av: AttributeValue) -> str:
    """true / false (the page states absence) / not_stated. Never infers false."""
    if av.status == "stated_absent" or (av.status == "stated" and av.value is False):
        return "false"
    if av.status == "stated" and av.value is True:
        return "true"
    return "not_stated"


def csv_extras(p: NormalizedProduct, record: dict) -> dict[str, Any]:
    d = record.get("details") or {}
    list_price = p.price.value if p.price.status == "stated" else None
    if p.price_after_discount.status == "stated":
        current = p.price_after_discount.value
    else:
        current = list_price if p.availability.value != "Out of stock" else None
    return {
        "product_type": d.get("product_type", "single"),
        "handle_count": d.get("handle_count"),
        "list_price": list_price,
        "current_price": current,
        "price_as_of": (record.get("retrieved_at") or "")[:10],
        "availability_raw": p.availability.raw_value if p.availability.status == "stated" else None,
        "price_source": "embedded product data on the manufacturer store page",
        "source_snapshot": record.get("raw_html_path"),
        "text_snapshot": record.get("text_snapshot_path"),
        "content_hash": record.get("content_hash"),
        "source_variant_url": record.get("product_url"),
        "source_variant_sku": d.get("source_variant_sku"),
        "source_variant_label": d.get("source_variant_label"),
        "variant_values": d.get("variant_values") or {},
        "merchandising_tags": d.get("merchandising_tags") or [],
        "modes_canonical": canonicalize_modes(d.get("modes") or []),
    }


def build_rows(products: list[NormalizedProduct], records_by_id: dict[str, dict]) -> list[dict[str, str]]:
    """CSV-ready rows (strings) for every column, CSV_ONLY_COLUMNS included."""
    rows = []
    for p in products:
        flat = p.to_flat_dict()
        flat.update(csv_extras(p, records_by_id[p.product_id]))
        row = {col: _csv_cell(col, flat.get(col)) for col in CSV_COLUMNS + CSV_ONLY_COLUMNS}
        for col in FLAG_COLUMNS:
            row[col] = flag_cell(getattr(p, col))
        if p.availability.status != "stated":
            row["availability"] = "not_stated"  # the visible page shows no stock cue
        rows.append(row)
    groups = alias_groups([{**r} for r in rows], sorted(FINGERPRINT_COLUMNS & set(rows[0].keys())))
    for r in rows:
        g = groups[r["product_id"]]
        r["alias_group"], r["canonical_product_id"] = g["alias_group"], g["canonical_product_id"]
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: the BOM makes Excel decode the file as UTF-8 (otherwise
    # curly apostrophes etc. show up as mojibake); other readers ignore it.
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv(products: list[NormalizedProduct], path: Path, records_by_id: dict[str, dict] | None = None) -> None:
    """Wide review table. ``records_by_id`` supplies the audit columns."""
    if records_by_id is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for p in products:
                flat = p.to_flat_dict()
                writer.writerow([_csv_cell(col, flat.get(col)) for col in CSV_COLUMNS])
        return
    _write_csv(path, CSV_COLUMNS + CSV_ONLY_COLUMNS, build_rows(products, records_by_id))


def write_parquet(products: list[NormalizedProduct], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list] = {col: [] for col in CSV_COLUMNS}
    for p in products:
        flat = p.to_flat_dict()
        for col in CSV_COLUMNS:
            value = flat.get(col)
            if col in JSON_ENCODED_COLUMNS and value is not None:
                value = json.dumps(value, ensure_ascii=False)
            columns[col].append(value)
    pq.write_table(pa.table(columns), path)


def write_evidence(evidence_items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence_items, indent=2, ensure_ascii=False), encoding="utf-8")


def write_warnings(warnings: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(warnings, indent=2, ensure_ascii=False), encoding="utf-8")


# --- documentation tables ----------------------------------------------------------------------------

COLUMN_DOC: dict[str, tuple[str, str]] = {
    "product_id": ("Stable id of the product row (brand + model slug).", "identity"),
    "brand": ("Philips or Oral-B.", "identity"),
    "product_name": ("Product title as the live page renders it.", "identity"),
    "product_url": ("Final (post-redirect) US product URL the values were read from.", "identity"),
    "discovered_url": ("URL as found on the listing page before redirects.", "provenance"),
    "retrieved_at": ("UTC timestamp of the page snapshot; prices/promos are valid as of this date.", "provenance"),
    "price": ("REGULAR (list) price in USD - the price before any promotion. Use current_price for 'how much is it now'. Blank when the page shows no price (out of stock).", "evidence"),
    "price_raw": ("Verbatim price text/number the price was read from.", "evidence"),
    "list_price": ("Alias of price (regular price).", "derived (CSV only)"),
    "current_price": ("Price after any discount (price_after_discount, else the regular price). Blank when no price is displayed. Answer 'how much is it' from this column.", "derived (CSV only)"),
    "original_price": ("Regular price as listed (same value as price).", "evidence"),
    "price_after_discount": ("Price after promotion; equals price when there is no discount.", "evidence"),
    "discount_amount": ("original_price - price_after_discount.", "evidence"),
    "discount_pct": ("Discount as a percentage of original_price.", "evidence"),
    "promotion_text": ("Verbatim promotion text shown on the page.", "evidence"),
    "price_as_of": ("Date of the snapshot the price comes from.", "provenance (CSV only)"),
    "availability_raw": ("The site's own wording behind `availability`: the visible buy-box label (\"Add to cart ($329.99)\" or \"Find A Retailer\"). \"Out of stock\" is our reading of \"Find A Retailer\", not text on the page. Applies to the colour variant in source_variant_label.", "provenance (CSV only)"),
    "price_source": ("Where the price number comes from (manufacturer store page data).", "provenance (CSV only)"),
    "availability": ("In stock (visible \"Add to cart\" button), Out of stock (no add-to-cart, \"Find A Retailer\" shown) or not_stated (no visible cue). Status of the scraped colour variant; the page's own wording is in availability_raw.", "evidence"),
    "listed_price_hidden": ("A price present in page data but not displayed to shoppers (out of stock). Never cited.", "evidence (cross-check)"),
    "currency": ("ISO currency of the price.", "evidence"),
    "price_type": ("listed / unavailable.", "evidence"),
    "mode_count": ("Number of brushing modes: stated count, else len(modes).", "evidence"),
    "modes": ("Mode names as written on the page.", "evidence"),
    "mode_details": ("Mode names with the page's description of each.", "evidence"),
    "modes_canonical": ("Each mode mapped to a canonical key with equivalence exact|approximate (e.g. Whitening ~ whiten approximate; Sensitive and Super Sensitive stay separate).", "derived (CSV only)"),
    "pressure_sensing": ("true if the page states a pressure sensor/feedback; not_stated otherwise (absence is never inferred).", "evidence, tri-state"),
    "app_connectivity": ("true if the page states an app connection.", "evidence, tri-state"),
    "bluetooth_connectivity": ("true only if the visible page text says Bluetooth (hidden merchandising tags do not count).", "evidence, tri-state"),
    "timer": ("true if the page states a brushing timer.", "evidence, tri-state"),
    "coverage_tracking": ("true if the product tells you which mouth zones you have covered (position detection, 'tracks how and where you brush', real-time coverage).", "evidence, tri-state"),
    "replacement_reminder": ("true only if a DEVICE feature signals/tracks when to replace the brush head. Advice such as 'replace every 3 months' does not count.", "evidence, tri-state"),
    "travel_case_included": ("true if the what's-in-the-box list contains a travel case/pouch.", "evidence, tri-state"),
    "display_type": ("A display or indicator as the page names it. A light ring is a pressure indicator unless the page calls it a display.", "evidence"),
    "cleaning_technology": ("The page's own sentence naming the cleaning/motor technology (micro-vibrations, oscillating-rotating, Sonicare/sonic technology, bristle movements). not_stated if the page does not say; never inferred from other products.", "evidence"),
    "brush_head_design": ("The page's own sentence about the brush head design (e.g. 'dentist-inspired round brush head').", "evidence"),
    "battery_life_days": ("Battery runtime in days (weeks x 7).", "evidence"),
    "battery_life_raw": ("Verbatim battery runtime text.", "evidence"),
    "battery_life_note": ("Oral-B: verbatim battery text (may be non-numeric).", "evidence"),
    "charging_method": ("magnetic / charger, only when the visible page states it.", "evidence"),
    "included_items": ("What's-in-the-box items as listed on the page.", "evidence"),
    "claims": ("Verbatim comparative/marketing claims; every claim also appears as a row of claims_long.", "evidence"),
    "plaque_removal_claim": ("Best plaque claim sentence; ALL distinct plaque claims are in claims_long.", "evidence"),
    "gum_health_claim": ("Best gum-health claim sentence; all in claims_long.", "evidence"),
    "disclaimers": ("Product footnotes with the claims each one qualifies.", "evidence"),
    "claims_with_disclaimers": ("Claims with their footnote text and in-sentence qualifiers.", "evidence"),
    "specifications": ("Real specification rows (Philips spec table; Oral-B dimensions/weight). Marketing headlines are in feature_headlines.csv.", "evidence"),
    "product_type": ("single | bundle. Bundles (2+ handles, Twin/Dual/Duo pack) are excluded from the table.", "derived (CSV only)"),
    "handle_count": ("Number of toothbrush handles in the box (brush heads are not counted).", "derived (CSV only)"),
    "alias_group": ("Rows whose page content is >=95% identical share a group (iO9 and iO9 Genius).", "derived (CSV only)"),
    "canonical_product_id": ("The group's base product (iO9 for iO9 Genius).", "derived (CSV only)"),
    "source_snapshot": ("Saved raw HTML of the page.", "provenance (CSV only)"),
    "text_snapshot": ("Saved cleaned text snapshot evidence offsets refer to.", "provenance (CSV only)"),
    "content_hash": ("SHA-256 of the saved HTML.", "provenance (CSV only)"),
    "source_variant_url": ("The colour/SKU page the row was read from.", "provenance (CSV only)"),
    "source_variant_sku": ("SKU of that variant page.", "provenance (CSV only)"),
    "source_variant_label": ("Colour/variant label of that page.", "provenance (CSV only)"),
    "variant_values": ("Numeric facts that differ between variant pages: {field: {sku: value}}. Empty when variants agree or were not compared.", "provenance (CSV only)"),
    "merchandising_tags": ("Hidden merchandising text on the page (gift-card blurb, TextFeatureN). Never evidence, never cited.", "provenance (CSV only)"),
    "variants": ("Other colour/SKU pages of the same model.", "provenance"),
}


def write_column_dictionary(path: Path, columns: list[str]) -> None:
    rows = []
    for c in columns:
        meaning, kind = COLUMN_DOC.get(c, ("", ""))
        if not meaning:
            if c.endswith("_details"):
                meaning, kind = ("Verbatim sentences about this topic from the visible page text.", "evidence")
            else:
                meaning, kind = ("Attribute read from the page; blank/not_stated when the page does not state it.", "evidence")
        rows.append({"column": c, "meaning": meaning, "kind": kind,
                     "in_parquet": "no" if c in CSV_ONLY_COLUMNS else "yes"})
    _write_csv(path, ["column", "meaning", "kind", "in_parquet"], rows)


def write_column_coverage(path: Path, columns: list[str], rows: list[dict]) -> None:
    out = []
    for c in columns:
        filled = sum(1 for r in rows if r.get(c, "").strip() and r.get(c) != "not_stated")
        out.append({"column": c, "filled": filled, "total": len(rows), "fill_rate": f"{filled / len(rows):.0%}"})
    _write_csv(path, ["column", "filled", "total", "fill_rate"], out)


def print_summary(products, evidence_count, warnings, errors, gate_warnings, trace) -> None:
    brand_counts = Counter(p.brand for p in products)
    warning_counts = Counter(w["code"] for w in warnings)
    print()
    print(f"Products processed: {len(products)}")
    for brand in sorted(brand_counts):
        print(f"{brand}: {brand_counts[brand]}")
    print(f"Evidence records: {evidence_count}  (offsets located: {trace['located']}, missing: {trace['missing_offset']}, cross_check_only: {trace['cross_check_only']})")
    print(f"Warnings: {len(warnings)}")
    for code, count in sorted(warning_counts.items()):
        print(f"  - {code}: {count}")
    print()
    print("Attribute status coverage:")
    status_counts: dict[str, Counter] = {f: Counter() for f in ATTRIBUTE_FIELDS}
    for p in products:
        for field_name in ATTRIBUTE_FIELDS:
            status_counts[field_name][getattr(p, field_name).status] += 1
    for field_name in ATTRIBUTE_FIELDS:
        c = status_counts[field_name]
        print(
            f"  {field_name:26s} stated={c['stated']:2d}  not_mentioned={c['not_mentioned']:2d}  "
            f"stated_absent={c['stated_absent']:2d}  ambiguous={c['ambiguous']:2d}"
        )
    print()
    print(f"Validation: {'FAIL' if errors else 'PASS'}")
    for err in errors:
        print(f"  ERROR   {err}")
    print(f"Quality-gate warnings: {len(gate_warnings)}")
    for w in gate_warnings[:60]:
        print(f"  warning {w}")
    if len(gate_warnings) > 60:
        print(f"  ... {len(gate_warnings) - 60} more (run scripts/validate_table.py)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    configure_logging(args.verbose)

    records = load_raw_records(RAW_DIR)
    if not records:
        logger.error("No raw records found under %s -- run scripts/build_data.py first.", RAW_DIR)
        return 1

    manifest_path = RAW_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_total = manifest.get("totals", {}).get("scraped")
        if expected_total is not None and expected_total != len(records):
            logger.warning(
                "manifest.json reports %d scraped products but %d raw JSON records were loaded from disk.",
                expected_total, len(records),
            )

    products, store, warning_objs = normalize_all(records)
    raw_index = index_raw_records(records)
    validation_errors = validate(products, store, raw_index)

    # evidence traceability (snapshot offsets, visibility, tier)
    trace = annotate_evidence(store, raw_index, RAW_DIR, PROCESSED_DIR / "snapshots")
    evidence_items = store.to_json_list()
    warnings = [w.model_dump() for w in warning_objs]

    rows = build_rows(products, raw_index)
    all_columns = CSV_COLUMNS + CSV_ONLY_COLUMNS
    _write_csv(PROCESSED_DIR / "comparison_table.csv", all_columns, rows)
    write_parquet(products, PROCESSED_DIR / "comparison_table.parquet")
    write_evidence(evidence_items, PROCESSED_DIR / "evidence.json")
    write_warnings(warnings, PROCESSED_DIR / "normalization_warnings.json")

    _write_csv(PROCESSED_DIR / "claims_long.csv", [
        "row_id", "product_id", "claim_text", "attribute", "also_in", "topic", "figure", "figure_value", "figure_unit",
        "comparator", "timeframe", "condition", "marker", "disclaimer_text", "conflict", "evidence_id", "source_section", "visibility",
        "evidence_tier", "snapshot_kind", "snapshot_file", "source_offset",
    ], build_claims_long(products, store))
    _write_csv(PROCESSED_DIR / "specs_long.csv", [
        "row_id", "product_id", "group", "name", "value", "unit", "qualifier", "item_text", "value_raw", "origin",
        "evidence_id", "source_section", "visibility",
        "evidence_tier", "snapshot_kind", "snapshot_file", "source_offset",
    ], build_specs_long(products, store))
    headlines = build_feature_headlines(raw_index)
    headline_trace = annotate_headlines(headlines, raw_index, RAW_DIR, PROCESSED_DIR / "snapshots")
    print(f"feature_headlines: {headline_trace['located']} located, {headline_trace['missing']} without an offset")
    _write_csv(PROCESSED_DIR / "feature_headlines.csv", [
        "row_id", "product_id", "evidence_id", "headline", "kind", "source_section", "visibility", "evidence_tier",
        "snapshot_kind", "snapshot_file", "source_offset", "quote_form",
    ], headlines)
    _write_csv(PROCESSED_DIR / "included_items_long.csv", [
        "row_id", "product_id", "item", "item_raw", "quantity", "quantity_source", "kind", "evidence_id", "source_section", "visibility",
        "evidence_tier", "snapshot_kind", "snapshot_file", "source_offset",
    ], build_included_items_long(products, store))
    _write_csv(PROCESSED_DIR / "product_aliases.csv",
               ["alias", "normalized_alias", "alias_type", "product_id", "canonical_product_id", "ambiguous", "candidates"],
               build_aliases([
                   {"product_id": r["product_id"], "brand": r["brand"], "product_name": r["product_name"],
                    "product_url": r["product_url"], "canonical_product_id": r["canonical_product_id"]}
                   for r in rows
               ]))
    write_column_dictionary(PROCESSED_DIR / "column_dictionary.csv", all_columns)
    write_column_coverage(PROCESSED_DIR / "column_coverage.csv", all_columns, rows)

    gate_errors, gate_warnings = check_files(PROCESSED_DIR, RAW_DIR, ROOT)
    errors = validation_errors + gate_errors
    print_summary(products, len(evidence_items), warnings, errors, gate_warnings, trace)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
