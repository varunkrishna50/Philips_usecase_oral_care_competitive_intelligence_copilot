#!/usr/bin/env python3
"""Step 1 data-acquisition pipeline for the Philips vs. Oral-B electric
toothbrush competitive-intelligence copilot.

Discovers product pages from the two approved US listing pages, scrapes
each one, and writes raw JSON + HTML snapshots under data/raw/, plus a
data/raw/manifest.json discovery/acquisition report.

This stage does NOT normalize, embed, or interpret the acquired content.

Usage:
    python scripts/build_data.py [--refresh] [--target-per-brand N] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from acquisition import oralb, philips  # noqa: E402
from acquisition.models import ExcludedUrl, FailedUrl, ProductRecord  # noqa: E402
from acquisition.utils import RateLimiter  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
HTML_DIR = DATA_DIR / "html"
PHILIPS_RAW_DIR = DATA_DIR / "philips"
ORALB_RAW_DIR = DATA_DIR / "oralb"
PHILIPS_HTML_DIR = HTML_DIR / "philips"
ORALB_HTML_DIR = HTML_DIR / "oralb"

logger = logging.getLogger("acquisition")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(DATA_DIR / "acquisition.log", mode="a", encoding="utf-8"),
        ],
    )


def _brand_summary(
    discovered: list,
    records: list[ProductRecord],
    failed: list[FailedUrl],
    excluded: list[ExcludedUrl],
) -> dict:
    return {
        "discovered_count": len(discovered),
        "scraped_count": len(records),
        "failed_count": len(failed),
        "excluded_count": len(excluded),
        "discovered_urls": [d.product_url for d in discovered],
        "excluded_urls": [e.model_dump() for e in excluded],
        "scraped_urls": [r.product_url for r in records],
        "failed_urls": [f.model_dump() for f in failed],
    }


def build_manifest(
    philips_discovered, philips_records, philips_failed, philips_excluded,
    oralb_discovered, oralb_records, oralb_failed, oralb_excluded,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pages": {
            "philips": philips.CATEGORY_URL,
            "oralb": oralb.CATEGORY_URL,
        },
        "totals": {
            "discovered": len(philips_discovered) + len(oralb_discovered),
            "scraped": len(philips_records) + len(oralb_records),
            "failed": len(philips_failed) + len(oralb_failed),
        },
        "brands": {
            "Philips": _brand_summary(philips_discovered, philips_records, philips_failed, philips_excluded),
            "Oral-B": _brand_summary(oralb_discovered, oralb_records, oralb_failed, oralb_excluded),
        },
    }


def print_summary(manifest: dict) -> None:
    print()
    for brand in ("Philips", "Oral-B"):
        b = manifest["brands"][brand]
        print(f"{brand}:")
        print(f"  discovered: {b['discovered_count']}")
        print(f"  successfully scraped: {b['scraped_count']}")
        print(f"  failed: {b['failed_count']}")
        print(f"  excluded (accessories/bundles/non-US/etc.): {b['excluded_count']}")
        if b["excluded_urls"]:
            reason_counts: dict[str, int] = {}
            for e in b["excluded_urls"]:
                reason_counts[e["reason"]] = reason_counts.get(e["reason"], 0) + 1
            for reason, count in sorted(reason_counts.items()):
                print(f"    - {reason}: {count}")
        if b["failed_urls"]:
            for f in b["failed_urls"]:
                print(f"    FAILED: {f['url']} ({f['reason']})")
        print()
    print(f"Total successfully acquired: {manifest['totals']['scraped']}")
    print(f"Full discovery report (discovered/excluded/scraped/failed URLs): {DATA_DIR / 'manifest.json'}")


def _drop_duplicates(records: list[ProductRecord], brand: str) -> list[ProductRecord]:
    """One record per product: two discovered URLs can resolve to the same
    page (e.g. a renamed model whose old URL now redirects), so keep the
    first record per final URL and per output file."""
    kept: list[ProductRecord] = []
    seen_urls: set[str] = set()
    seen_files: set[str] = set()
    for record in records:
        url_key = record.product_url.rstrip("/").lower()
        file_key = Path(record.raw_html_path).stem
        if url_key in seen_urls or file_key in seen_files:
            logger.warning("%s: dropping duplicate record %s (already have %s)", brand, record.discovered_url, record.product_url)
            continue
        seen_urls.add(url_key)
        seen_files.add(file_key)
        kept.append(record)
    return kept


def _remove_stale_files(records: list[ProductRecord], raw_dir: Path, html_dir: Path, brand_key: str) -> None:
    """After a --refresh run, delete saved records/snapshots that no longer
    correspond to a current product (renamed, merged or removed pages)."""
    keep = {Path(r.raw_html_path).stem for r in records}
    for path in list(raw_dir.glob(f"{brand_key}_*.json")) + list(html_dir.glob(f"{brand_key}_*.html")):
        if path.stem.startswith("_listing_") or path.stem in keep:
            continue
        logger.warning("Removing stale %s file: %s", brand_key, path.name)
        path.unlink()


def run_philips(session: requests.Session, rate_limiter: RateLimiter, refresh: bool, target_count: int):
    logger.info("=== Discovering Philips products ===")
    discovered, excluded = philips.discover(session, rate_limiter, PHILIPS_HTML_DIR, refresh, target_count=target_count)
    logger.info("Philips: %d candidate products discovered, %d URLs excluded", len(discovered), len(excluded))

    records: list[ProductRecord] = []
    failed: list[FailedUrl] = []
    for item in discovered:
        record, failure = philips.scrape(item, session, rate_limiter, PHILIPS_RAW_DIR, PHILIPS_HTML_DIR, refresh)
        if record is not None:
            records.append(record)
            logger.info("Philips OK: %s", record.product_url)
        else:
            assert failure is not None
            failed.append(failure)
            logger.warning("Philips FAILED: %s (%s)", failure.url, failure.reason)
    records = _drop_duplicates(records, "Philips")
    return discovered, records, failed, excluded


def run_oralb(session: requests.Session, rate_limiter: RateLimiter, refresh: bool, target_count: int):
    logger.info("=== Discovering Oral-B products ===")
    discovered, excluded, discovery_meta = oralb.discover(session, rate_limiter, ORALB_HTML_DIR, refresh, target_count=target_count)
    logger.info("Oral-B: %d candidate products discovered, %d URLs excluded", len(discovered), len(excluded))

    records: list[ProductRecord] = []
    failed: list[FailedUrl] = []
    for item in discovered:
        meta = discovery_meta[item.product_url]
        record, failure = oralb.scrape(item, meta, session, rate_limiter, ORALB_RAW_DIR, ORALB_HTML_DIR, refresh)
        if record is not None:
            records.append(record)
            logger.info("Oral-B OK: %s", record.product_url)
        else:
            assert failure is not None
            failed.append(failure)
            logger.warning("Oral-B FAILED: %s (%s)", failure.url, failure.reason)
    records = _drop_duplicates(records, "Oral-B")
    return discovered, records, failed, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Force fresh retrieval, ignoring any cached raw pages.")
    parser.add_argument("--target-per-brand", type=int, default=10, help="Preferred number of distinct models to discover per brand (default: 10).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging(args.verbose)

    session = requests.Session()
    # Oral-B's robots.txt asks AI/data crawlers for a 10s crawl delay;
    # Philips imposes no such requirement but we stay polite.
    rate_limiter = RateLimiter(delays={"oralb.com": 10.0, "www.usa.philips.com": 2.0}, default_delay=2.0)

    philips_discovered, philips_records, philips_failed, philips_excluded = run_philips(
        session, rate_limiter, args.refresh, args.target_per_brand
    )
    oralb_discovered, oralb_records, oralb_failed, oralb_excluded = run_oralb(
        session, rate_limiter, args.refresh, args.target_per_brand
    )

    if args.refresh:
        for records, raw_dir, html_dir, brand_key in (
            (philips_records, PHILIPS_RAW_DIR, PHILIPS_HTML_DIR, "philips"),
            (oralb_records, ORALB_RAW_DIR, ORALB_HTML_DIR, "oralb"),
        ):
            if len(records) >= 8:  # never prune on the back of a mostly-failed run
                _remove_stale_files(records, raw_dir, html_dir, brand_key)

    manifest = build_manifest(
        philips_discovered, philips_records, philips_failed, philips_excluded,
        oralb_discovered, oralb_records, oralb_failed, oralb_excluded,
    )
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for brand, records in (("Philips", philips_records), ("Oral-B", oralb_records)):
        if len(records) < 8:
            logger.error(
                "%s: only %d products successfully scraped (below the 8-product minimum). "
                "Review data/raw/manifest.json failed_urls/excluded_urls before proceeding to Step 2.",
                brand,
                len(records),
            )

    print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
