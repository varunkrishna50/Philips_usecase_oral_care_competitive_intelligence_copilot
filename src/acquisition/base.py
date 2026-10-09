"""Brand-agnostic fetch orchestration: requests -> (optional) Playwright
fallback -> generic content extraction, plus the on-disk cache.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .models import ProductRecord
from .snapshot import snapshot_path, write_snapshot
from .utils import (
    DEFAULT_USER_AGENT,
    FetchResult,
    RateLimiter,
    clean_soup,
    extract_json_ld,
    extract_meta_tags,
    extract_sections,
    extract_visible_text,
    fetch_url,
    needs_js_fallback,
    sha256_hex,
    slugify,
)

logger = logging.getLogger("acquisition")


@dataclass
class FetchOutcome:
    html: str
    soup: BeautifulSoup
    raw_text: str
    sections: list[dict]
    structured_data: dict
    final_url: str
    redirect_chain: list[str]
    status: int
    method: str


def try_playwright_fetch(url: str, timeout_ms: int = 30000) -> Optional[FetchResult]:
    """Render a page with headless Chromium. Returns None (logged) if
    Playwright is not installed or rendering fails -- never raises, so a
    missing browser install degrades to a reported failure, not a crash.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error(
            "Playwright is not installed; cannot render %s. "
            "Install it with: pip install playwright && playwright install chromium",
            url,
        )
        return None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(user_agent=DEFAULT_USER_AGENT)
                response = page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                html = page.content()
                status = response.status if response is not None else 0
                final_url = page.url
            finally:
                browser.close()
        return FetchResult(html=html, final_url=final_url, status=status, redirect_chain=[], method="playwright")
    except Exception as exc:  # noqa: BLE001 - Playwright raises many exception types
        logger.error("Playwright fetch failed for %s: %s", url, exc)
        return None


def fetch_and_prepare(
    url: str,
    session: requests.Session,
    rate_limiter: RateLimiter,
) -> FetchOutcome:
    """Fetch a URL and extract generic content. Falls back to Playwright
    only if the plain HTTP response looks like it needed JS rendering.
    May raise BotProtectionDetected or requests.RequestException, which
    callers must handle as a failed URL (not a pipeline-stopping error).
    """
    result = fetch_url(url, session, rate_limiter)

    html = result.html
    final_url = result.final_url
    redirect_chain = result.redirect_chain
    status = result.status
    method = result.method

    # Structured metadata (JSON-LD, <meta> tags) must come from the raw,
    # unmodified DOM: clean_soup() strips <script>/<nav>/<footer> tags to
    # produce readable body text, which would also strip the very
    # <script type="application/ld+json"> blocks we need here.
    raw_soup = BeautifulSoup(html, "lxml")
    soup = clean_soup(html)
    raw_text = extract_visible_text(soup)

    if needs_js_fallback(soup, raw_text):
        logger.info("Response for %s looks JS-dependent; trying Playwright fallback", url)
        pw_result = try_playwright_fetch(url)
        if pw_result is not None:
            html = pw_result.html
            final_url = pw_result.final_url
            status = pw_result.status
            method = "playwright"
            raw_soup = BeautifulSoup(html, "lxml")
            soup = clean_soup(html)
            raw_text = extract_visible_text(soup)
        else:
            logger.warning("JS fallback unavailable/failed for %s; keeping static response", url)

    sections = extract_sections(soup)
    structured_data = {
        "json_ld": extract_json_ld(raw_soup),
        "meta_tags": extract_meta_tags(raw_soup),
    }

    return FetchOutcome(
        html=html,
        soup=soup,
        raw_text=raw_text,
        sections=sections,
        structured_data=structured_data,
        final_url=final_url,
        redirect_chain=redirect_chain,
        status=status,
        method=method,
    )


def prepare_outcome(
    html: str,
    final_url: str,
    status: int = 200,
    redirect_chain: list[str] | None = None,
    method: str = "requests_cache",
) -> FetchOutcome:
    """The post-fetch half of ``fetch_and_prepare`` for HTML already on disk
    (no request, no JS fallback): used to re-derive records offline."""
    raw_soup = BeautifulSoup(html, "lxml")
    soup = clean_soup(html)
    return FetchOutcome(
        html=html,
        soup=soup,
        raw_text=extract_visible_text(soup),
        sections=extract_sections(soup),
        structured_data={"json_ld": extract_json_ld(raw_soup), "meta_tags": extract_meta_tags(raw_soup)},
        final_url=final_url,
        redirect_chain=redirect_chain or [],
        status=status,
        method=method,
    )


def fetch_listing_page(
    url: str,
    session: requests.Session,
    rate_limiter: RateLimiter,
    html_dir: Path,
    refresh: bool,
) -> BeautifulSoup:
    """Fetch a category/listing page, cached on disk by URL slug.

    Returns the RAW (unstripped) parsed DOM, not the chrome-cleaned one:
    product grids on these sites can legitimately sit inside a <nav> or
    similar landmark, and discovery needs to see every <a> tag. Positive
    filtering (accessory/bundle/category-page exclusion) happens on URL
    pattern and page content, not by removing markup up front.
    """
    cache_path = html_dir / f"_listing_{slugify(url)}.html"
    if cache_path.exists() and not refresh:
        logger.info("Using cached listing page for %s", url)
        html = cache_path.read_text(encoding="utf-8")
        return BeautifulSoup(html, "lxml")

    outcome = fetch_and_prepare(url, session, rate_limiter)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(outcome.html, encoding="utf-8")
    return BeautifulSoup(outcome.html, "lxml")


def record_json_path(raw_dir: Path, brand: str, slug: str) -> Path:
    return raw_dir / f"{brand}_{slug}.json"


def record_html_path(html_dir: Path, brand: str, slug: str) -> Path:
    return html_dir / f"{brand}_{slug}.html"


def portable_html_path(html_path: Path, html_dir: Path) -> str:
    """Repo-relative POSIX path (data/raw/html/<brand>/<file>) so records
    stay readable on any machine; html_dir is <repo>/data/raw/html/<brand>.
    """
    try:
        return html_path.resolve().relative_to(html_dir.resolve().parents[3]).as_posix()
    except (ValueError, IndexError):
        return str(html_path)


def load_cached_record(raw_dir: Path, brand: str, slug: str) -> Optional[ProductRecord]:
    path = record_json_path(raw_dir, brand, slug)
    if not path.exists():
        return None
    try:
        record = ProductRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.details is None:
            logger.info("Cached record %s predates structured details; will re-fetch", path)
            return None
        return record
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load cached record %s (%s); will re-fetch", path, exc)
        return None


def save_record(record: ProductRecord, html_content: str, raw_dir: Path, html_dir: Path, brand_key: str, slug: str) -> None:
    """brand_key must be the same lowercase key used by load_cached_record
    (e.g. "philips"/"oralb"), NOT record.brand's human-readable form --
    those must match exactly or the on-disk cache silently stops hitting.
    """
    html_path = record_html_path(html_dir, brand_key, slug)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_content, encoding="utf-8")
    record.raw_html_path = portable_html_path(html_path, html_dir)

    # Cleaned text snapshot: what evidence offsets are measured against.
    text_path = snapshot_path(raw_dir.parent if raw_dir.name != "raw" else raw_dir, brand_key, f"{brand_key}_{slug}")
    write_snapshot(text_path, [s.model_dump() for s in record.sections])
    record.text_snapshot_path = portable_html_path(text_path, html_dir)

    json_path = record_json_path(raw_dir, brand_key, slug)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def compute_content_hash(html_content: str) -> str:
    return sha256_hex(html_content.encode("utf-8"))
