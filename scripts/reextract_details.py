#!/usr/bin/env python3
"""Re-derive each record's extracted fields from the HTML snapshots already
on disk -- no page requests. Use it after changing extraction logic in
src/acquisition/*.

  Oral-B  : rebuilds name, sections, raw_text and details from the saved HTML
            (same code path as a live scrape, via oralb.build_record).
  Philips : rebuilds details only.

Stored ratings are reused. ``--ratings`` re-fetches Oral-B ratings with a
headless browser (the Bazaarvoice widget only exists after JavaScript runs).
``retrieved_at`` is preserved: the snapshot is not newer than it was.

Usage:
    python scripts/reextract_details.py [--ratings]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bs4 import BeautifulSoup  # noqa: E402

from acquisition import oralb, philips  # noqa: E402
from acquisition.base import portable_html_path, prepare_outcome  # noqa: E402
from acquisition.models import DiscoveredProduct, ProductRecord  # noqa: E402
from acquisition.utils import RateLimiter, extract_json_ld  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


def _product_ld(html: str) -> list[dict]:
    return [b for b in extract_json_ld(BeautifulSoup(html, "lxml")) if b.get("@type") == "Product"]


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

            if brand_key == "philips":
                raw_soup = BeautifulSoup(html, "lxml")
                sku_full = slug.upper().replace("-", "_")
                record.details = philips._extract_details(html, raw_soup, sku_full, _product_ld(html))
            else:
                outcome = prepare_outcome(
                    html, record.product_url, record.http_status, record.redirect_chain, record.scrape_method
                )
                outcome.structured_data.update(
                    {k: v for k, v in record.structured_data.items() if k in ("price_text", "rating_widget_text")}
                )
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
                record = rebuilt

            record.raw_html_path = portable_html_path(html_path, html_dir)
            json_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            print(f"{json_path.name}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
