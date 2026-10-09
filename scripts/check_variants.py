#!/usr/bin/env python3
"""Variant awareness: do a product's other colour/SKU pages state the same numbers?

For every product this fetches AT MOST ONE sibling variant page (the first
entry of ``variants``; cached under data/raw/html/<brand>/variants/, throttled
like the main scraper) and compares numeric facts the pages state:

    regular price, price after discount, number of modes, number of intensity
    levels, brush movements, battery runtime (days)

If the two pages disagree, both values are stored with their SKUs in
``details.variant_values`` (``{field: {sku: value}}``), which the CSV exposes as
``variant_values`` and the quality gate reports as a warning. Nothing is
overwritten: the row keeps the values of the page it was scraped from
(``source_variant_*`` columns).

Usage:
    python scripts/check_variants.py [--refresh]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from acquisition import oralb, philips  # noqa: E402
from acquisition.details import js_json_parse_payload, unescape_embedded_json  # noqa: E402
from acquisition.utils import DEFAULT_USER_AGENT, RateLimiter, extract_json_ld  # noqa: E402

RAW = ROOT / "data" / "raw"


def _numbers_philips(html: str, sku_full: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    ld = [b for b in extract_json_ld(soup) if b.get("@type") == "Product"]
    d = philips._extract_details(html, soup, sku_full, ld)
    text = unescape_embedded_json(html)
    mv = re.search(r"([\d,]{4,})\s*(?:bristle|brush) movements", text)
    days = re.search(r"Operating time[^}]{0,200}?\"csValueName\":\"(\d+)\s*(day|week)", text)
    out = {
        "regular_price": d.original_price, "price_after_discount": d.sale_price,
        "mode_count": len(d.modes) or d.mode_count, "intensity_levels": len(d.intensities) or None,
        "brush_movements_per_min": int(mv.group(1).replace(",", "")) if mv else None,
        "battery_life_days": (int(days.group(1)) * (7 if days.group(2) == "week" else 1)) if days else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def _numbers_oralb(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    ld = [b for b in extract_json_ld(soup) if b.get("@type") == "Product"]
    d = oralb._extract_details(html, soup, ld, None, [])
    weeks = re.search(r"charge lasts (\d+)\s*(day|week)", html, re.I)
    out = {
        "regular_price": d.original_price, "price_after_discount": d.sale_price,
        "mode_count": d.mode_count,
        "battery_life_days": (int(weeks.group(1)) * (7 if weeks.group(2).lower() == "week" else 1)) if weeks else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    limiter = RateLimiter({"oralb.com": 10.0, "www.usa.philips.com": 2.0}, default_delay=2.0)
    session = requests.Session()
    session.headers["User-Agent"] = DEFAULT_USER_AGENT

    for brand_key in ("philips", "oralb"):
        for path in sorted((RAW / brand_key).glob(f"{brand_key}_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            details = record.get("details") or {}
            variants = record.get("variants") or []
            if not variants:
                print(f"{path.stem}: no variants")
                continue
            sibling = variants[0]["url"]
            own_sku = details.get("source_variant_sku") or path.stem
            sib_sku = variants[0].get("sku") or urlparse(sibling).path.rstrip("/").rsplit("/", 1)[-1]
            cache = RAW / "html" / brand_key / "variants" / f"{path.stem}__{re.sub(r'[^A-Za-z0-9]+', '-', sib_sku)}.html"
            if cache.exists() and not args.refresh:
                sib_html = cache.read_text(encoding="utf-8")
            else:
                limiter.wait(sibling)
                resp = session.get(sibling, timeout=30)
                if resp.status_code != 200:
                    print(f"{path.stem}: sibling {sib_sku} -> HTTP {resp.status_code}, skipped")
                    continue
                sib_html = resp.text
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(sib_html, encoding="utf-8")
            own_html = (ROOT / record["raw_html_path"]).read_text(encoding="utf-8")
            if brand_key == "philips":
                mine = _numbers_philips(own_html, own_sku)
                theirs = _numbers_philips(sib_html, sib_sku.replace("/", "_"))
            else:
                mine, theirs = _numbers_oralb(own_html), _numbers_oralb(sib_html)
            diffs = {
                k: {str(own_sku): mine[k], str(sib_sku): theirs[k]}
                for k in mine if k in theirs and mine[k] != theirs[k]
            }
            details["variant_values"] = diffs
            record["details"] = details
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{path.stem}: compared with {sib_sku}: {'DISAGREE ' + json.dumps(diffs) if diffs else 'same'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
