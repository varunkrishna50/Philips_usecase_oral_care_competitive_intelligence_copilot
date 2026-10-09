#!/usr/bin/env python3
"""Step 2: normalize Step 1's raw acquisition data into a grounded,
auditable comparison dataset.

Reads data/raw/ only -- never re-scrapes the websites. Writes:

    data/processed/comparison_table.csv
    data/processed/comparison_table.parquet
    data/processed/evidence.json
    data/processed/normalization_warnings.json

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

from normalization.models import ATTRIBUTE_FIELDS, NormalizedProduct  # noqa: E402
from normalization.normalize import index_raw_records, load_raw_records, normalize_all  # noqa: E402
from normalization.validation import validate  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

CSV_COLUMNS = [
    "product_id",
    "brand",
    "product_name",
    "product_url",
    "discovered_url",
    "source_listing_url",
    "retrieved_at",
    "price",
    "price_raw",
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
    "variants",
]

JSON_ENCODED_COLUMNS = {
    "modes", "included_items", "claims", "specifications", "variants",
    "colors", "mode_details", "intensity_levels", "disclaimers", "claims_with_disclaimers",
    "pressure_sensing_details", "app_connectivity_details", "bluetooth_details", "battery_details", "charging_details", "coverage_tracking_details", "intensity_details",
}

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
    if column in JSON_ENCODED_COLUMNS:
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_csv(products: list[NormalizedProduct], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: the BOM makes Excel decode the file as UTF-8 (otherwise
    # curly apostrophes etc. show up as "â€™"); other readers ignore it.
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for p in products:
            flat = p.to_flat_dict()
            writer.writerow([_csv_cell(col, flat.get(col)) for col in CSV_COLUMNS])


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
    table = pa.table(columns)
    pq.write_table(table, path)


def write_evidence(evidence_items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence_items, indent=2, ensure_ascii=False), encoding="utf-8")


def write_warnings(warnings: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(warnings, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(
    products: list[NormalizedProduct],
    evidence_count: int,
    warnings: list[dict],
    validation_errors: list[str],
) -> None:
    brand_counts = Counter(p.brand for p in products)
    warning_counts = Counter(w["code"] for w in warnings)

    print()
    print(f"Products processed: {len(products)}")
    for brand in sorted(brand_counts):
        print(f"{brand}: {brand_counts[brand]}")
    print(f"Evidence records: {evidence_count}")
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
            f"  {field_name:24s} stated={c['stated']:2d}  not_mentioned={c['not_mentioned']:2d}  "
            f"stated_absent={c['stated_absent']:2d}  ambiguous={c['ambiguous']:2d}"
        )

    print()
    print(f"Validation: {'FAIL' if validation_errors else 'PASS'}")
    if validation_errors:
        for err in validation_errors:
            print(f"  - {err}")


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
                expected_total,
                len(records),
            )

    products, store, warning_objs = normalize_all(records)
    raw_index = index_raw_records(records)

    validation_errors = validate(products, store, raw_index)

    evidence_items = store.to_json_list()
    warnings = [w.model_dump() for w in warning_objs]

    write_csv(products, PROCESSED_DIR / "comparison_table.csv")
    write_parquet(products, PROCESSED_DIR / "comparison_table.parquet")
    write_evidence(evidence_items, PROCESSED_DIR / "evidence.json")
    write_warnings(warnings, PROCESSED_DIR / "normalization_warnings.json")

    print_summary(products, len(evidence_items), warnings, validation_errors)

    return 1 if validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
