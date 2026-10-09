#!/usr/bin/env python3
"""Read-only drift report: refetch every product URL stored in data/raw and
compare with the saved record (redirects, 404s, name, regular price, sale
price, promotion). Nothing on disk is modified.

Prices and promotions change within days and products get renamed or merged,
so run this before reviewing the comparison table; if it reports drift, run
``python scripts/build_data.py --refresh`` and rebuild the table.

Usage:
    python scripts/check_freshness.py [--brand oralb|philips]

Exit status 0 = no drift, 1 = drift found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from acquisition import oralb, philips  # noqa: E402
from acquisition.details import js_json_parse_payload  # noqa: E402
from acquisition.models import ProductRecord  # noqa: E402
from acquisition.utils import DEFAULT_USER_AGENT, RateLimiter, extract_json_ld  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


def _live_oralb(html: str) -> dict:
    main = (js_json_parse_payload(html, "jscontext") or {}).get("mainProduct") or {}
    price = main.get("price") or {}
    sale = (price.get("without_tax") or {}).get("value")
    rrp = (price.get("rrp_without_tax") or {}).get("value")
    promo = oralb._custom_field_values(main, "promotionText")
    return {"name": main.get("title"), "regular": rrp if rrp is not None else sale, "sale": sale, "promo": promo[0] if promo else None}


def _live_philips(html: str, slug: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    ld = [b for b in extract_json_ld(soup) if b.get("@type") == "Product"]
    d = philips._extract_details(html, soup, slug.upper().replace("-", "_"), ld)
    return {"name": ld[0].get("name") if ld else None, "regular": d.original_price, "sale": d.sale_price, "promo": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--brand", choices=("oralb", "philips"))
    args = parser.parse_args()
    limiter = RateLimiter({"oralb.com": 10.0, "www.usa.philips.com": 2.0}, default_delay=2.0)
    session = requests.Session()
    session.headers["User-Agent"] = DEFAULT_USER_AGENT

    drift = 0
    for brand_key in ("philips", "oralb"):
        if args.brand and args.brand != brand_key:
            continue
        for path in sorted((RAW_DIR / brand_key).glob(f"{brand_key}_*.json")):
            record = ProductRecord.model_validate_json(path.read_text(encoding="utf-8"))
            slug = path.stem[len(brand_key) + 1:]
            limiter.wait(record.product_url)
            problems: list[str] = []
            try:
                resp = session.get(record.product_url, timeout=30)
            except requests.RequestException as exc:
                print(f"{path.stem}: UNREACHABLE ({exc})")
                drift += 1
                continue
            if resp.status_code != 200:
                problems.append(f"HTTP {resp.status_code}")
            if resp.history and resp.url.rstrip("/") != record.product_url.rstrip("/"):
                problems.append(f"redirects to {resp.url}")
            if resp.status_code == 200:
                live = _live_oralb(resp.text) if brand_key == "oralb" else _live_philips(resp.text, slug)
                d = record.details
                if d is not None:
                    if live["regular"] is not None and live["regular"] != d.original_price:
                        problems.append(f"regular price {d.original_price} -> {live['regular']}")
                    if live["sale"] is not None and live["sale"] != d.sale_price:
                        problems.append(f"sale price {d.sale_price} -> {live['sale']}")
                    if (live["promo"] or None) != (d.promotion_text or None):
                        problems.append("promotion changed")
                if live["name"] and live["name"].strip() != record.product_name.strip():
                    problems.append(f"name {record.product_name!r} -> {live['name']!r}")
            status = "OK" if not problems else "DRIFT: " + "; ".join(problems)
            drift += bool(problems)
            print(f"{path.stem[:70].ljust(70)} {status}")
    print(f"\n{drift} product(s) with drift")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
