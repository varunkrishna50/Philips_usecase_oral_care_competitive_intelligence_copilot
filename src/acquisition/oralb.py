"""Discovery and scraping logic specific to Oral-B (US).

Oral-B's listing pages show every color variant as its own URL (unlike
Philips, where the category grid shows one URL per model). So variant
grouping has to happen at discovery time, using a slug heuristic, and is
then cross-checked against the authoritative color-swatch links embedded
in the scraped product page itself.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

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

logger = logging.getLogger("acquisition.oralb")

BRAND = "oralb"
DOMAIN = "oralb.com"
CATEGORY_URL = "https://oralb.com/en-us/products/compare/electric-toothbrushes"

# Individual product-detail pages live under /en-us/products/electric-toothbrushes/<slug>/
# (they 301-redirect to the canonical /en-us/products/shop-all/<slug>/ URL).
PRODUCT_HREF_RE = re.compile(r"^/en-us/products/electric-toothbrushes/([a-z0-9][a-z0-9\-]*)/?$")
PAGINATION_PATH = "/en-us/products/electric-toothbrushes/electric-toothbrushes"

# Category/collection landing pages that share the product URL prefix but
# are not themselves a single product.
CATEGORY_LANDING_SLUGS = {
    "electric-toothbrushes",
    "genius-series",
    "pro-series",
    "smart-series",
    "twin-packs-and-bundles",
}
BUNDLE_KEYWORDS = ("twin-pack", "bundle")

# Paths excluded outright: brush heads, floss/water flossers, manual
# toothbrushes, and kids products (a separate product line/category not
# reachable from the approved electric-toothbrushes listing's own results).
ACCESSORY_PATH_PREFIXES = (
    "/en-us/products/replacement-brush-heads",
    "/en-us/products/more-products/floss",
    "/en-us/products/more-products/water-flosser",
    "/en-us/products/toothbrushes/manual-toothbrushes",
    "/en-us/products/toothbrushes/clic",
    "/en-us/products/kids/",
)

COLOR_TOKENS = {
    "aqua", "marine", "black", "onyx", "rose", "quartz", "white", "alabaster",
    "violet", "ametrine", "cosmic", "stardust", "grey", "gray", "opal", "matte",
    "sapphire", "blue", "azalea", "pink", "night", "forest", "green", "pure",
    "lava", "gold", "silver", "red", "orange", "teal", "clear", "ivory",
}


@dataclass
class DiscoveryMeta:
    group_key: str
    heuristic_sibling_urls: list[str] = field(default_factory=list)


def normalize_base_slug(slug: str) -> str:
    """Strip a leading 'oral-b' brand prefix and trailing color tokens so
    that URLs for different colors of the same model collapse to one key.
    Deliberately conservative: only known color words are stripped, so a
    materially different product (e.g. an added '...-genius-...' variant)
    keeps its own distinct base.
    """
    tokens = slug.split("-")
    if len(tokens) > 2 and tokens[0] == "oral" and tokens[1] == "b":
        tokens = tokens[2:]
    while len(tokens) > 1 and tokens[-1] in COLOR_TOKENS:
        tokens.pop()
    return "-".join(tokens)


def _slug_from_path(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def discover(
    session: requests.Session,
    rate_limiter: RateLimiter,
    html_dir: Path,
    refresh: bool,
    target_count: int = 10,
    max_listing_pages: int = 8,
) -> tuple[list[DiscoveredProduct], list[ExcludedUrl], dict[str, DiscoveryMeta]]:
    raw_candidates: dict[str, DiscoveredProduct] = {}
    excluded: list[ExcludedUrl] = []
    visited_listings: set[str] = set()
    queue: list[str] = [CATEGORY_URL]
    seen_excluded: set[str] = set()

    while queue and len(visited_listings) < max_listing_pages:
        listing_url = queue.pop(0)
        if listing_url in visited_listings:
            continue
        visited_listings.add(listing_url)
        logger.info("Fetching Oral-B listing page: %s", listing_url)

        try:
            soup = fetch_listing_page(listing_url, session, rate_limiter, html_dir, refresh)
        except (BotProtectionDetected, requests.RequestException) as exc:
            logger.error("Could not fetch Oral-B listing page %s: %s", listing_url, exc)
            continue

        next_listing_candidates: list[str] = []
        for a in soup.find_all("a", href=True):
            full = urljoin(listing_url, a["href"])
            parsed = urlparse(full)
            if parsed.netloc not in (DOMAIN, f"www.{DOMAIN}"):
                continue
            path = parsed.path

            # Pagination links on the listing itself share the exact same
            # path as the bare "electric-toothbrushes" landing page, just
            # with a ?page=N query string -- check for that *before* the
            # product-href match below, since that match would otherwise
            # claim this path first and misfile it as an excluded landing
            # page without ever seeing the query string.
            if path.rstrip("/") == PAGINATION_PATH and parsed.query.startswith("page="):
                page_url = f"https://{DOMAIN}{path}?{parsed.query}"
                next_listing_candidates.append(page_url)
                continue

            match = PRODUCT_HREF_RE.match(path)
            if match:
                slug = match.group(1)
                full_url = f"https://{DOMAIN}{path.rstrip('/')}/"
                if slug in CATEGORY_LANDING_SLUGS:
                    if full_url not in seen_excluded:
                        seen_excluded.add(full_url)
                        excluded.append(
                            ExcludedUrl(
                                brand="Oral-B",
                                url=full_url,
                                reason="category_or_collection_landing_page",
                                discovered_from=listing_url,
                            )
                        )
                    continue
                if any(keyword in slug for keyword in BUNDLE_KEYWORDS):
                    if full_url not in seen_excluded:
                        seen_excluded.add(full_url)
                        excluded.append(
                            ExcludedUrl(
                                brand="Oral-B",
                                url=full_url,
                                reason="twin_pack_or_bundle_excluded_by_acquisition_policy",
                                discovered_from=listing_url,
                            )
                        )
                    continue
                if slug not in raw_candidates:
                    name_guess = a.get_text(strip=True) or slug.replace("-", " ").title()
                    raw_candidates[slug] = DiscoveredProduct(
                        brand="Oral-B",
                        product_name=name_guess,
                        product_url=full_url,
                        discovered_from=listing_url,
                    )
                continue

            if path.startswith(ACCESSORY_PATH_PREFIXES):
                if full not in seen_excluded:
                    seen_excluded.add(full)
                    excluded.append(
                        ExcludedUrl(
                            brand="Oral-B",
                            url=full,
                            reason="accessory_manual_or_kids_path_excluded",
                            discovered_from=listing_url,
                        )
                    )
                continue

        for candidate in next_listing_candidates:
            if candidate not in visited_listings and candidate not in queue:
                queue.append(candidate)

        groups_so_far = {
            normalize_base_slug(_slug_from_path(urlparse(d.product_url).path))
            for d in raw_candidates.values()
        }
        if len(groups_so_far) >= target_count and not queue:
            break

    groups: dict[str, list[DiscoveredProduct]] = {}
    for item in raw_candidates.values():
        slug = _slug_from_path(urlparse(item.product_url).path)
        base = normalize_base_slug(slug)
        groups.setdefault(base, []).append(item)

    discovered: list[DiscoveredProduct] = []
    discovery_meta: dict[str, DiscoveryMeta] = {}
    for base, items in groups.items():
        items.sort(key=lambda d: d.product_url)
        representative = items[0]
        siblings = [d.product_url for d in items[1:]]
        discovered.append(representative)
        discovery_meta[representative.product_url] = DiscoveryMeta(group_key=base, heuristic_sibling_urls=siblings)

    return discovered, excluded, discovery_meta


def _extract_name(soup) -> str | None:
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        return text or None
    return None


def _extract_price_text(soup) -> str | None:
    el = soup.find(attrs={"aria-label": re.compile(r"^Price of the product", re.I)})
    if el:
        return el.get_text(strip=True)
    el = soup.find(class_=lambda c: bool(c) and "price--withoutTax" in c)
    if el:
        return el.get_text(strip=True)
    return None


def _extract_variant_links(soup, current_url: str) -> list[VariantRef]:
    current_path = urlparse(current_url).path
    current_slug = _slug_from_path(current_path)
    current_base = normalize_base_slug(current_slug)

    variants: list[VariantRef] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(current_url, a["href"])
        path = urlparse(full).path
        m = PRODUCT_HREF_RE.match(path)
        if not m:
            continue
        slug = m.group(1)
        if slug == current_slug or normalize_base_slug(slug) != current_base:
            continue
        url_norm = f"https://{DOMAIN}{path.rstrip('/')}/"
        if url_norm in seen:
            continue
        seen.add(url_norm)

        label = a.get("aria-label") or a.get_text(strip=True) or None
        if not label:
            img = a.find("img")
            if img and img.get("alt"):
                label = img["alt"]
        variants.append(VariantRef(url=url_norm, label=label, sku=None))
    return variants


def _merge_variants(authoritative: list[VariantRef], heuristic_urls: list[str], current_url: str) -> list[VariantRef]:
    def norm(u: str) -> str:
        return u.rstrip("/") + "/"

    seen = {norm(v.url) for v in authoritative}
    seen.add(norm(current_url))
    merged = list(authoritative)
    for url in heuristic_urls:
        if norm(url) in seen:
            continue
        seen.add(norm(url))
        merged.append(VariantRef(url=url, label=None, sku=None))
    return merged


def scrape(
    discovered: DiscoveredProduct,
    discovery_meta: DiscoveryMeta,
    session: requests.Session,
    rate_limiter: RateLimiter,
    raw_dir: Path,
    html_dir: Path,
    refresh: bool,
) -> tuple[ProductRecord | None, FailedUrl | None]:
    slug = discovery_meta.group_key

    if not refresh:
        cached = load_cached_record(raw_dir, BRAND, slug)
        if cached is not None:
            logger.info("Using cached record for %s", discovered.product_url)
            return cached, None

    logger.info("Fetching Oral-B product page: %s", discovered.product_url)
    try:
        outcome = fetch_and_prepare(discovered.product_url, session, rate_limiter)
    except BotProtectionDetected as exc:
        return None, FailedUrl(
            brand="Oral-B",
            url=discovered.product_url,
            source_listing_url=discovered.discovered_from,
            reason=f"bot_protection: {exc}",
        )
    except requests.RequestException as exc:
        return None, FailedUrl(
            brand="Oral-B",
            url=discovered.product_url,
            source_listing_url=discovered.discovered_from,
            reason=f"request_error: {exc}",
            http_status=getattr(getattr(exc, "response", None), "status_code", None),
        )

    errors: list[str] = []
    json_ld_products = [
        block for block in outcome.structured_data.get("json_ld", []) if block.get("@type") == "Product"
    ]

    product_name = json_ld_products[0].get("name") if json_ld_products else None
    if not product_name:
        product_name = _extract_name(outcome.soup)
        if product_name:
            errors.append("product_name_from_h1_no_json_ld_present")

    if not product_name:
        product_name = discovered.product_name
        errors.append("product_name_fell_back_to_listing_link_text")

    price_text = _extract_price_text(outcome.soup)
    has_price = price_text is not None
    if json_ld_products and not has_price:
        offers = json_ld_products[0].get("offers")
        has_price = isinstance(offers, dict) and offers.get("price") is not None

    if not has_price:
        return None, FailedUrl(
            brand="Oral-B",
            url=discovered.product_url,
            source_listing_url=discovered.discovered_from,
            reason="no_price_signal_found_not_treated_as_product_page",
            http_status=outcome.status,
        )

    if price_text:
        outcome.structured_data["price_text"] = price_text

    if outcome.status != 200:
        errors.append(f"non_200_http_status:{outcome.status}")

    authoritative_variants = _extract_variant_links(outcome.soup, outcome.final_url)
    variants = _merge_variants(authoritative_variants, discovery_meta.heuristic_sibling_urls, outcome.final_url)

    record = ProductRecord(
        brand="Oral-B",
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
