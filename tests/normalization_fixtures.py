"""Shared helper for building synthetic Step-1-shaped raw records in Step
2 tests. Writes a small, realistic HTML file to disk (under tmp_path) so
code paths that read raw_html_path (price JSON-LD snippet lookup,
evidence verbatim re-validation) behave exactly as they do in production,
rather than being special-cased for tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def make_raw_record(
    tmp_path: Path,
    *,
    brand: str = "Philips",
    product_name: str = "Test Toothbrush",
    slug: str = "test-toothbrush",
    sections: Optional[list[dict]] = None,
    raw_text: Optional[str] = None,
    json_ld_price=None,
    json_ld_currency: Optional[str] = "USD",
    json_ld_availability: Optional[str] = None,
    price_text: Optional[str] = None,
    variants: Optional[list[dict]] = None,
    product_url: Optional[str] = None,
    extra_json_ld_blocks: Optional[list[dict]] = None,
) -> dict:
    brand_key = "philips" if brand == "Philips" else "oralb"
    sections = sections or []
    if raw_text is None:
        raw_text = "\n".join(s["heading"] + "\n" + s["text"] for s in sections)
    if product_url is None:
        product_url = f"https://example.com/{brand_key}/{slug}"

    json_ld: list[dict] = []
    if json_ld_price is not None:
        offer: dict = {"@type": "Offer", "price": json_ld_price}
        if json_ld_currency is not None:
            offer["priceCurrency"] = json_ld_currency
        if json_ld_availability is not None:
            offer["availability"] = json_ld_availability
        json_ld.append(
            {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": product_name,
                "offers": offer,
            }
        )
    if extra_json_ld_blocks:
        json_ld.extend(extra_json_ld_blocks)

    structured_data: dict = {"json_ld": json_ld, "meta_tags": {}}
    if price_text is not None:
        structured_data["price_text"] = price_text
        if price_text not in raw_text:
            raw_text = raw_text + "\n" + price_text

    html_dir = tmp_path / "html" / brand_key
    html_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_dir / f"{brand_key}_{slug}.html"
    json_ld_script = "".join(
        f'<script type="application/ld+json">{json.dumps(block)}</script>' for block in json_ld
    )
    html_body = f"<html><head>{json_ld_script}</head><body>{raw_text}</body></html>"
    html_path.write_text(html_body, encoding="utf-8")

    return {
        "brand": brand,
        "product_name": product_name,
        "product_url": product_url,
        "discovered_url": product_url,
        "redirect_chain": [],
        "source_listing_url": f"https://example.com/{brand_key}/listing",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "http_status": 200,
        "scrape_method": "requests",
        "raw_html_path": str(html_path),
        "content_hash": "0" * 64,
        "raw_text": raw_text,
        "sections": sections,
        "structured_data": structured_data,
        "variants": variants or [],
        "errors": [],
    }


def section(heading: str, text: str) -> dict:
    return {"heading": heading, "text": text}
