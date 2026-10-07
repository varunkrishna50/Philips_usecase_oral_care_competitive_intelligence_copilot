"""Discovery and scraping logic specific to Philips Sonicare (US)."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from .base import (
    compute_content_hash,
    fetch_and_prepare,
    fetch_listing_page,
    load_cached_record,
    record_html_path,
    save_record,
)
from .models import DiscoveredProduct, ExcludedUrl, FailedUrl, ProductRecord, Section, VariantRef
from .utils import BotProtectionDetected, RateLimiter

logger = logging.getLogger("acquisition.philips")

BRAND = "philips"
DOMAIN = "www.usa.philips.com"
CATEGORY_URL = "https://www.usa.philips.com/c-m-pe/electric-toothbrushes/latest"

# Official Philips product-detail URLs look like /c-p/<SKU>_<SUFFIX>/<slug>.
PRODUCT_HREF_RE = re.compile(r"^/c-p/([A-Za-z0-9]+)_([A-Za-z0-9]+)/([a-z0-9][a-z0-9\-]*)/?$")
# Other electric-toothbrush sub-category pages worth also checking for more products.
SUBCATEGORY_HREF_RE = re.compile(r"^/c-m-pe/electric-toothbrushes(?:/[a-z0-9\-]+)?/(?:latest|all)/?$")

ACCESSORY_SLUG_KEYWORDS = (
    "brush-head",
    "brush-heads",
    "replacement",
    "charger",
    "travel-case",
    "toothbrush-heads",
)


def _sku_full(sku: str, suffix: str) -> str:
    return f"{sku}_{suffix}"


def sku_filename_slug(sku_full: str) -> str:
    return sku_full.lower().replace("_", "-")


def discover(
    session: requests.Session,
    rate_limiter: RateLimiter,
    html_dir: Path,
    refresh: bool,
    target_count: int = 10,
) -> tuple[list[DiscoveredProduct], list[ExcludedUrl]]:
    discovered: dict[str, DiscoveredProduct] = {}
    excluded: list[ExcludedUrl] = []
    visited_listings: set[str] = set()
    queue: list[str] = [CATEGORY_URL]

    while queue:
        listing_url = queue.pop(0)
        if listing_url in visited_listings:
            continue
        visited_listings.add(listing_url)
        logger.info("Fetching Philips listing page: %s", listing_url)

        try:
            soup = fetch_listing_page(listing_url, session, rate_limiter, html_dir, refresh)
        except (BotProtectionDetected, requests.RequestException) as exc:
            logger.error("Could not fetch Philips listing page %s: %s", listing_url, exc)
            continue

        next_listing_candidates: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            parsed = urlparse(href)
            if parsed.netloc and parsed.netloc != DOMAIN:
                continue
            path = parsed.path

            match = PRODUCT_HREF_RE.match(path)
            if match:
                sku, suffix, slug = match.groups()
                full_url = f"https://{DOMAIN}{path}"
                if any(keyword in slug for keyword in ACCESSORY_SLUG_KEYWORDS):
                    excluded.append(
                        ExcludedUrl(
                            brand="Philips",
                            url=full_url,
                            reason="url_slug_matches_accessory_keyword",
                            discovered_from=listing_url,
                        )
                    )
                    continue
                if slug not in discovered:
                    name_guess = a.get_text(strip=True) or slug.replace("-", " ").title()
                    discovered[slug] = DiscoveredProduct(
                        brand="Philips",
                        product_name=name_guess,
                        product_url=full_url,
                        discovered_from=listing_url,
                    )
                continue

            if SUBCATEGORY_HREF_RE.match(path):
                full_sub = f"https://{DOMAIN}{path}"
                if full_sub not in visited_listings:
                    next_listing_candidates.append(full_sub)

        if len(discovered) >= target_count:
            break
        for candidate in next_listing_candidates:
            if candidate not in visited_listings and candidate not in queue:
                queue.append(candidate)

    return list(discovered.values()), excluded


def _extract_variants(soup, current_url: str) -> list[VariantRef]:
    cur_match = PRODUCT_HREF_RE.match(urlparse(current_url).path)
    if not cur_match:
        return []
    cur_sku, cur_suffix, cur_slug = cur_match.groups()
    cur_sku_full = _sku_full(cur_sku, cur_suffix)

    variants: list[VariantRef] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        path = urlparse(a["href"]).path
        m = PRODUCT_HREF_RE.match(path)
        if not m:
            continue
        sku, suffix, slug = m.groups()
        sku_full = _sku_full(sku, suffix)
        if slug != cur_slug or sku_full == cur_sku_full or sku_full in seen:
            continue
        seen.add(sku_full)
        variants.append(VariantRef(url=f"https://{DOMAIN}{path}", sku=sku_full, label=None))
    return variants


def scrape(
    discovered: DiscoveredProduct,
    session: requests.Session,
    rate_limiter: RateLimiter,
    raw_dir: Path,
    html_dir: Path,
    refresh: bool,
) -> tuple[ProductRecord | None, FailedUrl | None]:
    cur_match = PRODUCT_HREF_RE.match(urlparse(discovered.product_url).path)
    if not cur_match:
        return None, FailedUrl(
            brand="Philips",
            url=discovered.product_url,
            source_listing_url=discovered.discovered_from,
            reason="url_does_not_match_philips_product_pattern",
        )
    sku, suffix, _slug = cur_match.groups()
    slug = sku_filename_slug(_sku_full(sku, suffix))

    if not refresh:
        cached = load_cached_record(raw_dir, BRAND, slug)
        if cached is not None:
            logger.info("Using cached record for %s", discovered.product_url)
            return cached, None

    logger.info("Fetching Philips product page: %s", discovered.product_url)
    try:
        outcome = fetch_and_prepare(discovered.product_url, session, rate_limiter)
    except BotProtectionDetected as exc:
        return None, FailedUrl(
            brand="Philips",
            url=discovered.product_url,
            source_listing_url=discovered.discovered_from,
            reason=f"bot_protection: {exc}",
        )
    except requests.RequestException as exc:
        return None, FailedUrl(
            brand="Philips",
            url=discovered.product_url,
            source_listing_url=discovered.discovered_from,
            reason=f"request_error: {exc}",
            http_status=getattr(getattr(exc, "response", None), "status_code", None),
        )

    errors: list[str] = []
    json_ld_products = [
        block for block in outcome.structured_data.get("json_ld", []) if block.get("@type") == "Product"
    ]

    product_name = None
    has_price = False
    if json_ld_products:
        product_name = json_ld_products[0].get("name")
        offers = json_ld_products[0].get("offers")
        if isinstance(offers, dict) and offers.get("price") is not None:
            has_price = True

    if not product_name:
        h1 = outcome.soup.find("h1")
        product_name = h1.get_text(strip=True) if h1 else discovered.product_name
        errors.append("product_name_not_found_in_json_ld_used_h1_or_listing_text_fallback")

    if not has_price:
        return None, FailedUrl(
            brand="Philips",
            url=discovered.product_url,
            source_listing_url=discovered.discovered_from,
            reason="no_price_offer_found_not_treated_as_product_page",
            http_status=outcome.status,
        )

    if outcome.status != 200:
        errors.append(f"non_200_http_status:{outcome.status}")

    variants = _extract_variants(outcome.soup, discovered.product_url)

    record = ProductRecord(
        brand="Philips",
        product_name=product_name,
        product_url=outcome.final_url,
        discovered_url=discovered.product_url,
        redirect_chain=outcome.redirect_chain,
        source_listing_url=discovered.discovered_from,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        http_status=outcome.status,
        scrape_method=outcome.method,
        raw_html_path=str(record_html_path(html_dir, BRAND, slug)),
        content_hash=compute_content_hash(outcome.html),
        raw_text=outcome.raw_text,
        sections=[Section(**s) for s in outcome.sections],
        structured_data=outcome.structured_data,
        variants=variants,
        errors=errors,
    )
    save_record(record, outcome.html, raw_dir, html_dir, BRAND, slug)
    return record, None
