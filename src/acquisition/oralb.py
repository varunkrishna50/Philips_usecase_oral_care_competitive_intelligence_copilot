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
from bs4 import BeautifulSoup

from .base import (
    compute_content_hash,
    fetch_and_prepare,
    fetch_listing_page,
    load_cached_record,
    record_html_path,
    save_record,
)
from .details import (
    PRICE_TOLERANCE,
    compute_discount,
    extract_footnotes,
    extract_modes_from_lines,
    find_stated_mode_count,
    html_to_lines,
    js_json_parse_payload,
    parse_product_copy,
    split_box_item,
)
from .models import (
    DiscoveredProduct,
    Disclaimer,
    ExcludedUrl,
    Feature,
    FailedUrl,
    MarkedStatement,
    ProductDetails,
    ProductRecord,
    Section,
    VariantRef,
)
from .utils import DEFAULT_USER_AGENT, BotProtectionDetected, RateLimiter

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


def _first_number(text: str | None) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text or "")
    return float(m.group(0).replace(",", "")) if m else None


def _custom_field_values(main_product: dict, name: str) -> list[str]:
    return [
        str(f.get("value", "")).strip()
        for f in main_product.get("custom_fields", [])
        if f.get("name") == name and str(f.get("value", "")).strip()
    ]


def _clean_color_label(label: str) -> str:
    label = re.sub(r"^io\s*series[- ]*\d+[- ]*", "", label.strip(), flags=re.I).strip(" -")
    return label.title() if label.islower() else label


def _extract_colors(raw_soup) -> list[str]:
    """Color names exactly as the PDP swatch selector lists them. The
    selector is shared by a model family (e.g. the iO 9 and iO 9 Genius
    pages show the same swatches), so this is the number of colors the page
    offers, not necessarily the colors of this one SKU."""
    box = raw_soup.find(id="color-swatches-pdp")
    if box is None:
        return []
    colors: list[str] = []
    for a in box.find_all("a", href=True):
        img = a.find("img")
        label = (img.get("alt") or img.get("title") or "") if img else a.get_text(strip=True)
        label = _clean_color_label(label)
        if label and label not in colors:
            colors.append(label)
    return colors


def _extract_details(
    html: str,
    raw_soup,
    json_ld_products: list[dict],
    price_text: str | None,
    variants: list[VariantRef],
) -> ProductDetails:
    details = ProductDetails()
    notes = details.notes
    payload = js_json_parse_payload(html, "jscontext")
    main = (payload or {}).get("mainProduct")
    if not isinstance(main, dict):
        notes.append("embedded_product_json_not_found")
        main = {}

    # --- prices -----------------------------------------------------------
    price = main.get("price") or {}
    sale = (price.get("without_tax") or {}).get("value")
    rrp = (price.get("rrp_without_tax") or {}).get("value")
    saved = (price.get("saved") or {}).get("value")
    currency = (price.get("without_tax") or {}).get("currency")
    if sale is not None:
        details.price_source = "embedded_json:mainProduct.price"
    else:
        sale = _first_number(price_text)
        if sale is not None:
            details.price_source = "dom:price_text"
            notes.append("price_from_dom_text_embedded_json_missing")
    details.sale_price = float(sale) if sale is not None else None
    details.original_price = float(rrp) if rrp is not None else details.sale_price
    details.currency = currency
    details.discount_amount, details.discount_pct = compute_discount(details.original_price, details.sale_price)
    if saved is not None and details.original_price is not None and details.sale_price is not None:
        if abs(details.original_price - details.sale_price - float(saved)) > PRICE_TOLERANCE:
            notes.append(f"saved_amount_mismatch:rrp-sale={details.original_price - details.sale_price:.2f},saved={saved}")
    if json_ld_products and details.sale_price is not None:
        offers = json_ld_products[0].get("offers")
        ld_price = offers.get("price") if isinstance(offers, dict) else None
        try:
            if ld_price is not None and abs(float(ld_price) - details.sale_price) > PRICE_TOLERANCE:
                notes.append(f"json_ld_price_differs:{ld_price}")
        except (TypeError, ValueError):
            pass
    promo = _custom_field_values(main, "promotionText")
    details.promotion_text = promo[0] if promo else None

    # --- availability / displayed price / lead paragraph ---------------------
    avail = None
    for block in json_ld_products:
        offers = block.get("offers")
        if isinstance(offers, dict) and offers.get("availability"):
            avail = str(offers["availability"]).rsplit("/", 1)[-1]
    if avail:
        details.availability = {"InStock": "in_stock", "OutOfStock": "out_of_stock"}.get(avail, avail)
    if main.get("can_purchase") is False and details.availability in (None, "in_stock"):
        notes.append("can_purchase_false_but_json_ld_in_stock")
    if details.availability == "out_of_stock" or main.get("can_purchase") is False:
        details.availability = "out_of_stock"
    # Visible price block: absent on the live page when the product cannot be bought.
    details.price_displayed = details.availability != "out_of_stock"
    short = raw_soup.select_one("div.short-description")
    if short is not None:
        details.short_description = re.sub(r"\s+", " ", short.get_text(" ", strip=True)) or None

    # --- colors -----------------------------------------------------------
    details.colors = _extract_colors(raw_soup)
    if details.colors:
        details.color_count = len(details.colors)
    elif not variants:
        details.color_count = 1
        notes.append("color_count_1_no_swatch_selector_or_sibling_variants")
    else:
        notes.append("color_swatches_not_found")

    # --- product copy (shopper-visible: description bullets + tabs) -------------
    copy = parse_product_copy(main.get("warranty") or "")
    intro = copy["intro"]
    intro_lines = [line for line, _ in intro]
    details.highlights = [
        line for line in intro_lines
        if 25 <= len(line) <= 400 and not re.match(r"^\(\d+\)|^\*|^Do not|^Full T&Cs", line)
    ][:12]
    # Features: one per description bullet, named by its bold lead-in when the
    # page has one, otherwise by the first clause.
    for line, strong in intro:
        if len(line) < 25 or re.match(r"^\(\d+\)|^\*|^Do not|^Full T&Cs", line):
            continue
        name = strong.strip() or re.split(r"[:.,;—–]", line, maxsplit=1)[0].strip()
        details.features.append(Feature(name=name[:80], description=line))
    details.features = details.features[:12]

    for _, strong in intro:
        term = strong.strip()
        if 3 <= len(term) <= 60 and term not in details.key_terms:
            details.key_terms.append(term)
    for line in intro_lines:
        if re.search(r"(?<!\*)\*+", line) and not re.match(r"^\*", line):
            details.marked_statements.append(MarkedStatement(label="Product description", text=line))

    for heading, lines in copy["tabs"].items():
        for line in lines:
            if re.search(r"(?<!\*)\*+", line) and not re.match(r"^\*", line):
                details.marked_statements.append(MarkedStatement(label=heading, text=line))

    # What's in the box: the tab's list items, dimensions/weight split out.
    box_lines = copy["tabs"].get("What's in the box", [])
    for line in box_lines:
        item, dims, weight = split_box_item(line)
        if item and item not in details.box_items:
            details.box_items.append(item)
        details.dimensions = details.dimensions or dims
        details.weight = details.weight or weight
    if not details.box_items:
        notes.append("no_whats_in_the_box_list_found")

    # Disclaimers: footnote lines in the description or tabs.
    footnotes = extract_footnotes(intro_lines, "Product description")
    for heading, lines in copy["tabs"].items():
        footnotes += extract_footnotes(lines, heading)
    for fn in footnotes:
        source = "product" if fn["section"] == "Product description" else "section"
        details.disclaimers.append(Disclaimer(marker=fn["marker"], text=fn["text"], source=source, section=fn["section"]))

    # Modes: only from visible copy (description bullets, FAQ answers).
    faq_lines: list[str] = []
    for ans in raw_soup.select(".faq-answer"):
        faq_lines.extend(html_to_lines(str(ans)))
    stated, modes = extract_modes_from_lines(intro_lines + faq_lines)
    details.mode_count = stated
    details.modes = modes
    if stated is None and not modes:
        notes.append("no_mode_information_found")
    elif not modes:
        notes.append("mode_names_not_listed_on_page")
    elif stated is not None and len(modes) != stated:
        notes.append(f"mode_count_mismatch:stated={stated},parsed={len(modes)}")
    return details


def fetch_bazaarvoice_rating(
    url: str, rate_limiter: RateLimiter, timeout_ms: int = 45000
) -> tuple[float | None, int | None, str | None, str | None]:
    """Average rating and review count from the Bazaarvoice summary widget,
    which only exists after JavaScript runs. Returns (avg, count, error,
    widget_text); widget_text is the rendered text the numbers came from,
    kept so the rating has a verbatim source (it is not in the static HTML)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, None, "rating_unavailable_playwright_not_installed", None
    rate_limiter.wait(url)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                # en-US locale: without it oralb.com geo-redirects to another storefront.
                context = browser.new_context(locale="en-US", user_agent=DEFAULT_USER_AGENT)
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                if urlparse(page.url).path.split("/")[1:2] != ["en-us"]:
                    return None, None, f"rating_unavailable_redirected_to:{page.url}", None
                try:
                    page.wait_for_selector('[data-bv-show="rating_summary"][data-bv-ready="true"]', timeout=15000)
                except Exception:  # noqa: BLE001 - widget may legitimately not render (no reviews)
                    return None, None, "rating_widget_did_not_render", None
                texts = page.locator('[data-bv-show="rating_summary"]').all_inner_texts()
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001
        return None, None, f"rating_fetch_failed:{exc}", None

    for text in texts:
        m = re.search(r"(\d(?:\.\d+)?)\s*\|?\s*(?:([\d,]+)\s+Reviews?)?", text.strip())
        if m and m.group(2):
            return float(m.group(1)), int(m.group(2).replace(",", "")), None, text.strip()
    if any(re.search(r"No reviews", t, re.I) for t in texts):
        return None, 0, None, None
    return None, None, f"rating_text_not_parsed:{texts[:1]}", None


def embedded_copy_sections(main_product: dict) -> tuple[list[Section], str]:
    """The shopper-visible copy that only exists inside the embedded product
    JSON (tab bodies are rendered client-side) as headed sections, plus the
    same text as one block to append to raw_text so every extractor and the
    verbatim-evidence validator can see it."""
    copy = parse_product_copy(main_product.get("warranty") or "")
    sections: list[Section] = []
    block: list[str] = []
    intro_lines = [line for line, _ in copy["intro"]]
    if intro_lines:
        sections.append(Section(heading="Product description", text="\n".join(intro_lines)))
        block += ["Product description"] + intro_lines
    for heading, lines in copy["tabs"].items():
        if lines:
            sections.append(Section(heading=heading, text="\n".join(lines)))
            block += [heading] + lines
    return sections, "\n".join(block)


def _custom_fields(main_product: dict) -> list[dict]:
    return [
        {"name": str(f.get("name", "")), "value": str(f.get("value", ""))}
        for f in main_product.get("custom_fields", [])
    ]


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


def _is_not_found_page(outcome, html: str) -> bool:
    if outcome.status == 404:
        return True
    m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
    return bool(m and re.search(r"not found", m.group(1), re.I))


def _scrape_url(
    discovered: DiscoveredProduct,
    discovery_meta: DiscoveryMeta,
    session: requests.Session,
    rate_limiter: RateLimiter,
    raw_dir: Path,
    html_dir: Path,
    refresh: bool,
    prior_failure: str | None = None,
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

    if _is_not_found_page(outcome, outcome.html):
        return None, FailedUrl(
            brand="Oral-B",
            url=discovered.product_url,
            source_listing_url=discovered.discovered_from,
            reason="page_not_found_404",
            http_status=outcome.status,
        )

    return build_record(
        outcome, discovered, discovery_meta, rate_limiter, raw_dir, html_dir,
        prior_failure=prior_failure,
    )


def build_record(
    outcome,
    discovered: DiscoveredProduct,
    discovery_meta: DiscoveryMeta,
    rate_limiter: RateLimiter | None,
    raw_dir: Path,
    html_dir: Path,
    prior_failure: str | None = None,
    rating: tuple[float | None, int | None, str | None, str | None] | None = None,
    save: bool = True,
) -> tuple[ProductRecord | None, FailedUrl | None]:
    """Everything after the page is fetched: identity, price, details,
    sections. ``rating`` lets offline re-extraction reuse a stored rating
    instead of launching a browser; ``save=False`` skips writing files."""
    slug = discovery_meta.group_key
    errors: list[str] = []
    if prior_failure:
        errors.append(prior_failure)
    json_ld_products = [
        block for block in outcome.structured_data.get("json_ld", []) if block.get("@type") == "Product"
    ]
    payload = js_json_parse_payload(outcome.html, "jscontext") or {}
    main_product = payload.get("mainProduct") if isinstance(payload.get("mainProduct"), dict) else {}

    # Name precedence: the embedded product JSON is what the live page renders
    # (its <h1>); JSON-LD / <title> / og:url can lag behind after a rename.
    product_name = (main_product.get("title") or "").strip() or None
    ld_name = json_ld_products[0].get("name") if json_ld_products else None
    if product_name and ld_name and ld_name.strip() != product_name:
        errors.append(f"json_ld_name_differs_from_page_title:{ld_name}")
    if not product_name:
        product_name = _extract_name(outcome.soup)
        if product_name:
            errors.append("product_name_from_h1_embedded_product_json_missing")
    if not product_name:
        product_name = ld_name
        if product_name:
            errors.append("product_name_from_json_ld_only")
    if not product_name:
        product_name = discovered.product_name
        errors.append("product_name_fell_back_to_listing_link_text")

    price_text = _extract_price_text(outcome.soup)
    has_price = price_text is not None
    if json_ld_products and not has_price:
        offers = json_ld_products[0].get("offers")
        has_price = isinstance(offers, dict) and offers.get("price") is not None
    if not has_price and main_product:
        has_price = (main_product.get("price") or {}).get("without_tax", {}).get("value") is not None

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
    # Internal merchandising fields: kept for reference, never used as evidence.
    outcome.structured_data["custom_fields"] = _custom_fields(main_product)

    if outcome.status != 200:
        errors.append(f"non_200_http_status:{outcome.status}")

    # The URL may have been merged into another product since discovery:
    # file the record under the slug of the URL actually served.
    final_slug = normalize_base_slug(_slug_from_path(urlparse(outcome.final_url).path))
    if final_slug != slug:
        errors.append(f"redirected_group_{slug}_to_{final_slug}")
        slug = final_slug

    authoritative_variants = _extract_variant_links(outcome.soup, outcome.final_url)
    variants = _merge_variants(authoritative_variants, discovery_meta.heuristic_sibling_urls, outcome.final_url)

    details = _extract_details(
        outcome.html,
        BeautifulSoup(outcome.html, "lxml"),
        json_ld_products,
        price_text,
        variants,
    )
    if rating is None:
        rating = fetch_bazaarvoice_rating(outcome.final_url, rate_limiter)
    rating_avg, rating_count, rating_error, rating_text = rating
    details.rating_avg, details.rating_count = rating_avg, rating_count
    if rating_text:
        outcome.structured_data["rating_widget_text"] = rating_text
    if rating_error:
        details.notes.append(rating_error)

    extra_sections, extra_text = embedded_copy_sections(main_product)
    sections = [Section(**sec) for sec in outcome.sections] + extra_sections
    raw_text = outcome.raw_text + ("\n" + extra_text if extra_text else "")

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
        raw_text=raw_text,
        sections=sections,
        structured_data=outcome.structured_data,
        variants=variants,
        details=details,
        errors=errors,
    )
    if save:
        save_record(record, outcome.html, raw_dir, html_dir, BRAND, slug)
    return record, None


def scrape(
    discovered: DiscoveredProduct,
    discovery_meta: DiscoveryMeta,
    session: requests.Session,
    rate_limiter: RateLimiter,
    raw_dir: Path,
    html_dir: Path,
    refresh: bool,
) -> tuple[ProductRecord | None, FailedUrl | None]:
    """Scrape one model. If the representative color's page is gone (404 /
    not a product page), fall back to the model's other color URLs so a
    discontinued color does not drop the whole model."""
    candidates = [discovered.product_url] + [
        u for u in discovery_meta.heuristic_sibling_urls if u != discovered.product_url
    ]
    first_failure: FailedUrl | None = None
    for url in candidates:
        candidate = discovered.model_copy(update={"product_url": url})
        note = (
            f"representative_url_failed_used_sibling:{first_failure.url}:{first_failure.reason}"
            if first_failure is not None
            else None
        )
        record, failure = _scrape_url(
            candidate, discovery_meta, session, rate_limiter, raw_dir, html_dir, refresh, prior_failure=note
        )
        if record is not None:
            return record, None
        first_failure = first_failure or failure
        if failure is None or not failure.reason.startswith(("page_not_found", "no_price_signal")):
            break
    return None, first_failure
