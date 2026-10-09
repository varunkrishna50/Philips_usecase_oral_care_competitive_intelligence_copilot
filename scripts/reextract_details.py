#!/usr/bin/env python3
"""Re-derive each record's extracted fields from the HTML snapshots already
on disk -- no page requests. Use it after changing extraction logic in
src/acquisition/*.

  Both brands: sections / raw_text are rebuilt from the saved HTML with the
  current cleaner (widget removal, visibility tags), the cleaned text
  snapshot (data/raw/text/<brand>/<id>.txt) is rewritten, and details are
  re-extracted. Oral-B goes through the same ``oralb.build_record`` as a
  live scrape.
  Bundles (2+ handles, "Twin/Dual/Duo pack") are excluded by policy: their
  record, HTML and text snapshot are deleted and the manifest's excluded
  list is updated.

Stored ratings are reused. ``--ratings`` re-fetches Oral-B ratings with a
headless browser (the Bazaarvoice widget only exists after JavaScript runs).
``retrieved_at`` is preserved: the snapshot is not newer than it was.

Usage:
    python scripts/reextract_details.py [--ratings]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bs4 import BeautifulSoup  # noqa: E402

from acquisition import oralb, philips  # noqa: E402
from acquisition.base import portable_html_path, prepare_outcome  # noqa: E402
from acquisition.details import classify_bundle  # noqa: E402
from acquisition.models import DiscoveredProduct, ProductRecord, Section  # noqa: E402
from acquisition.snapshot import snapshot_path, write_snapshot  # noqa: E402
from acquisition.utils import RateLimiter, extract_json_ld  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
BRAND_LABEL = {"philips": "Philips", "oralb": "Oral-B"}


def _product_ld(html: str) -> list[dict]:
    return [b for b in extract_json_ld(BeautifulSoup(html, "lxml")) if b.get("@type") == "Product"]


def _exclude_bundle(brand_key: str, record: ProductRecord, json_path: Path, html_path: Path, reason: str) -> None:
    """Delete the bundle's files and record it as excluded in the manifest."""
    for path in (json_path, html_path, snapshot_path(RAW_DIR, brand_key, json_path.stem)):
        if path.exists():
            path.unlink()
    manifest_path = RAW_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        brand = manifest["brands"][BRAND_LABEL[brand_key]]
        entry = {
            "brand": BRAND_LABEL[brand_key],
            "url": record.product_url,
            "reason": f"bundle_excluded_by_policy: {reason}",
            "discovered_from": record.source_listing_url,
        }
        if not any(e["url"] == entry["url"] for e in brand["excluded_urls"]):
            brand["excluded_urls"].append(entry)
            brand["excluded_count"] = len(brand["excluded_urls"])
        brand["scraped_urls"] = [u for u in brand["scraped_urls"] if u != record.product_url]
        brand["scraped_count"] = len(brand["scraped_urls"])
        manifest["totals"]["scraped"] = sum(b["scraped_count"] for b in manifest["brands"].values())
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ratings", action="store_true", help="fetch Oral-B ratings with a headless browser")
    args = parser.parse_args()
    limiter = RateLimiter({"oralb.com": 10.0}, default_delay=2.0)

    for brand_key in ("philips", "oralb"):
        html_dir = RAW_DIR / "html" / brand_key
        for json_path in sorted((RAW_DIR / brand_key).glob(f"{brand_key}_*.json")):
            record = ProductRecord.model_validate_json(json_path.read_text(encoding="utf-8"))
            slug = json_path.stem[len(brand_key) + 1:]
            html_path = html_dir / f"{brand_key}_{slug}.html"
            html = html_path.read_text(encoding="utf-8")
            outcome = prepare_outcome(
                html, record.product_url, record.http_status, record.redirect_chain, record.scrape_method
            )
            keep = {k: v for k, v in record.structured_data.items() if k in ("price_text", "rating_widget_text", "custom_fields")}
            outcome.structured_data.update(keep)

            if brand_key == "philips":
                raw_soup = BeautifulSoup(html, "lxml")
                sku_full = slug.upper().replace("-", "_")
                record.sections = [Section(**s) for s in outcome.sections]
                record.raw_text = outcome.raw_text
                record.details = philips._extract_details(html, raw_soup, sku_full, _product_ld(html))
            else:
                old = record.details
                rating = None
                if not args.ratings:
                    rating = (
                        old.rating_avg if old else None,
                        old.rating_count if old else None,
                        None,
                        record.structured_data.get("rating_widget_text"),
                    )
                discovered = DiscoveredProduct(
                    brand="Oral-B", product_name=record.product_name,
                    product_url=record.product_url, discovered_from=record.source_listing_url,
                )
                meta = oralb.DiscoveryMeta(group_key=slug, heuristic_sibling_urls=[v.url for v in record.variants])
                rebuilt, failure = oralb.build_record(
                    outcome, discovered, meta, limiter, RAW_DIR / brand_key, html_dir, rating=rating, save=False
                )
                if rebuilt is None:
                    print(f"{json_path.name}: FAILED {failure.reason if failure else '?'}")
                    continue
                rebuilt.retrieved_at = record.retrieved_at
                rebuilt.discovered_url = record.discovered_url
                rebuilt.content_hash = record.content_hash
                # keep what separate browser/variant runs added to the old record
                rebuilt.rendered_text_path = record.rendered_text_path
                if old is not None and rebuilt.details is not None and old.variant_values:
                    rebuilt.details.variant_values = old.variant_values
                record = rebuilt

            ptype, handles, reason = classify_bundle(record.product_name, record.details.box_items if record.details else [])
            if record.details is not None:
                record.details.product_type, record.details.handle_count = ptype, handles
            if ptype == "bundle":
                _exclude_bundle(brand_key, record, json_path, html_path, reason or "bundle")
                print(f"{json_path.name}: EXCLUDED bundle ({reason})")
                continue

            record.raw_html_path = portable_html_path(html_path, html_dir)
            text_path = snapshot_path(RAW_DIR, brand_key, json_path.stem)
            write_snapshot(text_path, [s.model_dump() for s in record.sections])
            record.text_snapshot_path = portable_html_path(text_path, html_dir)
            json_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            print(f"{json_path.name}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
