"""Discovery and scraping logic specific to Philips Sonicare (US)."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .base import (
    compute_content_hash,
    fetch_and_prepare,
    fetch_listing_page,
    load_cached_record,
    record_html_path,
    save_record,
)
from .details import PRICE_TOLERANCE, compute_discount, decode_json_at, find_stated_mode_count, unescape_embedded_json
from .models import (
    DiscoveredProduct,
    Disclaimer,
    ExcludedUrl,
    Feature,
    FailedUrl,
    MarkedStatement,
    Mode,
    ProductDetails,
    ProductRecord,
    Section,
    VariantRef,
)
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


def _money(window: str, key: str) -> float | None:
    m = re.search(r'"' + key + r'":\{[^{}]*?"value":(-?[\d.]+)', window)
    return float(m.group(1)) if m else None


def _spec_chapters(text: str) -> dict[str, dict]:
    """First occurrence of each specification chapter, keyed by chapter name."""
    chapters: dict[str, dict] = {}
    for m in re.finditer(r'\{"csChapterCode"', text):
        chapter = decode_json_at(text, m.start())
        if isinstance(chapter, dict) and chapter.get("csChapterName") not in chapters:
            chapters[chapter.get("csChapterName")] = chapter
    return chapters


def _chapter_items(chapter: dict | None) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for item in (chapter or {}).get("csItem", []):
        values = [str(v.get("csValueName", "")).strip() for v in item.get("csValue", [])]
        out.append((str(item.get("csItemName", "")).strip(), [v for v in values if v]))
    return out


def _extract_details(html: str, raw_soup, sku_full: str, json_ld_products: list[dict]) -> ProductDetails:
    details = ProductDetails()
    notes = details.notes
    text = unescape_embedded_json(html)
    ld = json_ld_products[0] if json_ld_products else {}

    # --- prices -----------------------------------------------------------
    commerce_code = sku_full.replace("_", "/")
    start = text.find('"baseCommerceProduct":{"id":"' + commerce_code + '"')
    if start < 0:
        notes.append("embedded_price_block_not_found")
    window = text[start:start + 3000] if start >= 0 else ""
    list_price = _money(window, "price")
    discount_price = _money(window, "discountPrice")
    msrp = _money(window, "msrpPrice")
    cur = re.search(r'"currency":"([A-Z]{3})"', window)
    details.currency = cur.group(1) if cur else None
    offers = ld.get("offers") if isinstance(ld.get("offers"), dict) else {}
    ld_price = offers.get("price")
    if list_price is not None or discount_price is not None:
        details.price_source = "embedded_json:baseCommerceProduct"
        original = max(v for v in (list_price, msrp) if v is not None) if (list_price or msrp) else discount_price
        sale = discount_price if discount_price is not None else list_price
    else:
        original = sale = float(ld_price) if ld_price is not None else None
        details.price_source = "json_ld:offers.price" if sale is not None else None
        details.currency = details.currency or offers.get("priceCurrency")
        notes.append("price_from_json_ld_embedded_json_missing")
    details.original_price, details.sale_price = original, sale
    details.discount_amount, details.discount_pct = compute_discount(original, sale)
    try:
        if ld_price is not None and sale is not None and abs(float(ld_price) - sale) > PRICE_TOLERANCE:
            notes.append(f"json_ld_price_differs:{ld_price}")
    except (TypeError, ValueError):
        pass

    status = re.search(r'"availability":\{"status":"([A-Z_]+)"', window)
    if status:
        details.availability = {"IN_STOCK": "in_stock", "OUT_OF_STOCK": "out_of_stock"}.get(status.group(1), status.group(1).lower())
    details.price_displayed = True

    # --- specification chapters (modes, intensities, colour) --------------
    chapters = _spec_chapters(text)
    mode_count_from_value: int | None = None
    for name, values in _chapter_items(chapters.get("Modes")):
        if re.match(r"^\d+\s+intensit", name, re.I):
            details.intensities.extend(values)
        elif values and all(v.isdigit() for v in values):
            mode_count_from_value = int(values[0])
        else:
            details.modes.append(Mode(name=name, description="; ".join(values) or None))
    for name, values in _chapter_items(chapters.get("Intensities")):
        if name not in details.intensities:
            details.intensities.append(name)

    # --- highlights and features -----------------------------------------
    hl = re.search(r'"versions":\[((?:"[^"]*"(?:,)?)+)\]', text)
    if hl:
        details.highlights = [h.replace("\\u0026", "&") for h in re.findall(r'"([^"]*)"', hl.group(1))]
    items_at = text.find('"items":[{"featureLongDescription"')
    items = decode_json_at(text, items_at + len('"items":')) if items_at >= 0 else None
    if isinstance(items, list):
        for item in sorted(items, key=lambda i: int(i.get("featureRank") or 999)):
            name = str(item.get("featureName") or item.get("featureLongDescription") or "").strip()
            desc = str(item.get("featureGlossary") or item.get("featureLongDescription") or "").strip()
            if name:
                details.features.append(Feature(name=name, description=desc or None))
    else:
        notes.append("feature_list_not_found")

    # --- mode count, cross-checked against the marketing highlight --------
    highlight_count = find_stated_mode_count(details.highlights)
    if details.modes:
        details.mode_count = len(details.modes)
    elif mode_count_from_value is not None:
        details.mode_count = mode_count_from_value
        notes.append("mode_names_not_listed_on_page")
    elif highlight_count is not None:
        details.mode_count = highlight_count
        notes.append("mode_count_from_highlight_only")
    else:
        notes.append("no_mode_information_found")
    if highlight_count is not None and details.mode_count not in (None, highlight_count):
        notes.append(f"mode_count_mismatch:spec={details.mode_count},highlight={highlight_count}")

    # --- colors -----------------------------------------------------------
    label = next((p for p in raw_soup.find_all("p") if p.get_text(strip=True) == "Available in"), None)
    if label is not None:
        for a in label.parent.find_all("a", href=True):
            img = a.find("img")
            color = ((img.get("alt") if img else None) or a.get_text(strip=True)).strip()
            if color and color not in details.colors:
                details.colors.append(color)
    if details.colors:
        details.color_count = len(details.colors)
    else:
        spec_colors = [
            v for chapter in chapters.values() for n, vals in _chapter_items(chapter)
            if re.fullmatch(r"colou?rs?", n, re.I) for v in vals
        ]
        details.colors = list(dict.fromkeys(spec_colors))
        details.color_count = len(details.colors) or 1
        notes.append("color_from_spec_table" if spec_colors else "color_count_1_no_color_selector")

    # --- disclaimers: product-specific footnotes keyed by marker run -------
    # (the page's other "disclaimers" arrays are CMS blocks shared across
    # unrelated product categories and are deliberately ignored).
    at = text.find('"productDisclaimers":[')
    items_ = decode_json_at(text, at + len('"productDisclaimers":')) if at >= 0 else None
    if isinstance(items_, list):
        for item in sorted(items_, key=lambda i: int(i.get("rank") or 0)):
            body = str(item.get("disclaimerText") or "").strip()
            marker = str(item.get("marker") or "").strip()
            if body:
                details.disclaimers.append(Disclaimer(marker=marker, text=body, source="product"))

    # Statements (spec values, feature texts) that carry a footnote marker.
    marker_run = re.compile(r"(?<!\*)\*+")
    for cname, chapter in chapters.items():
        for iname, values in _chapter_items(chapter):
            for value in values:
                if marker_run.search(value):
                    details.marked_statements.append(
                        MarkedStatement(label=f"{cname}: {iname}", text=value)
                    )
    if isinstance(items, list):
        for item in items:
            for key in ("featureLongDescription", "featureGlossary"):
                value = str(item.get(key) or "")
                if marker_run.search(value):
                    details.marked_statements.append(
                        MarkedStatement(label=f"Feature: {item.get('featureName') or ''}".strip(), text=value)
                    )

    # --- rating: average and count only ----------------------------------
    agg = ld.get("aggregateRating")
    if isinstance(agg, dict):
        try:
            details.rating_avg = float(agg["ratingValue"])
            details.rating_count = int(agg["reviewCount"])
            details.rating_scale = float(agg.get("bestRating") or 5)
        except (KeyError, TypeError, ValueError):
            notes.append("aggregate_rating_unparseable")
    else:
        notes.append("no_aggregate_rating_on_page")
    return details


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
    details = _extract_details(
        outcome.html, BeautifulSoup(outcome.html, "lxml"), _sku_full(sku, suffix), json_ld_products
    )

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
        details=details,
        errors=errors,
    )
    save_record(record, outcome.html, raw_dir, html_dir, BRAND, slug)
    return record, None
