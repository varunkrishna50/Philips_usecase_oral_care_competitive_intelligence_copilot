"""Shared, brand-agnostic helpers: HTTP fetching, rate limiting, caching,
hashing, and generic HTML extraction.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("acquisition")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; ProductsResearch/1.0; research-data-acquisition)"
)

# Tags that are page chrome / non-content, removed before text & section
# extraction. This is "cleaning" raw markup noise, not normalizing claims.
BOILERPLATE_TAGS = ["script", "style", "noscript", "nav", "footer", "svg", "iframe"]

# CSS selectors for blocks that are in the DOM but never shown to a shopper
# (verified with a headless browser: is_visible() is False). Oral-B's
# BigCommerce template dumps internal merchandising fields here
# (TextFeatureN, "Shop By Feature", algolia-*, bvProductId, SKU/UPC...).
# They are not product copy, so they must not become evidence; the parsed
# fields are kept separately in structured_data["custom_fields"].
HIDDEN_METADATA_SELECTORS = ["dl.productView-info"]

HEADING_TAGS = ["h1", "h2", "h3", "h4"]
TEXT_TAGS = ["p", "li", "dd", "dt", "span", "td", "th", "blockquote"]

BOT_PROTECTION_MARKERS = (
    "attention required",
    "just a moment",
    "checking your browser",
    "access denied",
    "are you a human",
    "captcha",
    "pardon the interruption",
    "request unsuccessful",
)


class BotProtectionDetected(RuntimeError):
    """Raised when a response looks like a CAPTCHA / bot-protection page.

    This stage must never try to bypass this -- it is a hard stop for the
    affected URL, surfaced to the caller as a distinct failure reason.
    """


@dataclass
class FetchResult:
    html: str
    final_url: str
    status: int
    redirect_chain: list[str] = field(default_factory=list)
    method: str = "requests"


class RateLimiter:
    """Enforces a minimum delay between requests to the same domain."""

    def __init__(self, delays: dict[str, float], default_delay: float = 1.0):
        self._delays = delays
        self._default_delay = default_delay
        self._last_request: dict[str, float] = {}

    def wait(self, url: str) -> None:
        domain = urlparse(url).netloc
        delay = self._delays.get(domain, self._default_delay)
        last = self._last_request.get(domain)
        if last is not None:
            elapsed = time.monotonic() - last
            remaining = delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request[domain] = time.monotonic()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def looks_like_bot_protection(html: str, status: int) -> bool:
    if status in (403, 429, 503):
        lowered = html.lower()
        if any(marker in lowered for marker in BOT_PROTECTION_MARKERS):
            return True
        # A very short body on an error status is also suspicious.
        if len(html.strip()) < 2000:
            return True
    lowered_title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
    if lowered_title_match:
        title = lowered_title_match.group(1).lower()
        if any(marker in title for marker in BOT_PROTECTION_MARKERS):
            return True
    return False


def fetch_url(
    url: str,
    session: requests.Session,
    rate_limiter: RateLimiter,
    timeout: float = 20.0,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> FetchResult:
    """GET a URL with retries on transient errors, honoring rate limits.

    Raises BotProtectionDetected if the response looks like a CAPTCHA /
    bot-protection page. Raises requests.RequestException if the URL could
    not be fetched after retries.
    """
    last_exc: Optional[Exception] = None
    last_status: Optional[int] = None
    for attempt in range(1, max_retries + 1):
        rate_limiter.wait(url)
        try:
            response = session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Fetch attempt %d/%d failed for %s: %s", attempt, max_retries, url, exc)
            time.sleep(backoff_seconds * attempt)
            continue

        if looks_like_bot_protection(response.text, response.status_code):
            raise BotProtectionDetected(
                f"Bot protection / CAPTCHA detected at {url} (status {response.status_code})"
            )

        if response.status_code >= 500 or response.status_code == 429:
            last_status = response.status_code
            logger.warning(
                "Fetch attempt %d/%d got HTTP %s for %s, retrying",
                attempt,
                max_retries,
                response.status_code,
                url,
            )
            time.sleep(backoff_seconds * attempt)
            continue

        redirect_chain = [r.url for r in response.history]
        return FetchResult(
            html=response.text,
            final_url=response.url,
            status=response.status_code,
            redirect_chain=redirect_chain,
            method="requests",
        )

    if last_exc is not None:
        raise last_exc
    raise requests.HTTPError(
        f"Exhausted {max_retries} retries for {url}; last HTTP status: {last_status}"
    )


def clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for tag_name in BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for selector in HIDDEN_METADATA_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    return soup


def extract_visible_text(soup: BeautifulSoup) -> str:
    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


# Elements that start a new line of text. Everything else (a, span, strong,
# em, b, i, ...) is inline and is joined to the text around it.
BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "body", "button", "dd", "details", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
    "li", "main", "ol", "option", "p", "pre", "section", "summary", "table", "tbody", "td", "tfoot",
    "th", "thead", "tr", "ul",
})


def extract_sections(soup: BeautifulSoup) -> list[dict]:
    """Group the page's text under the most recent heading, in document order.

    Text is split into lines at block-element boundaries and <br>: the text of
    one block (including its inline children such as <strong>/<a>/<span>) is
    one line, and text that sits directly inside a container (e.g. a bare
    text node in a <ul>) is not lost. This avoids both the fragments and the
    duplicates that come from treating every nested tag as its own line.
    Driven by tag semantics only, not nth-child position.
    """
    from bs4 import Comment, NavigableString, Tag

    sections: list[dict] = []
    state = {"heading": "Page content", "lines": []}
    heading_names = set(HEADING_TAGS)

    def flush_section() -> None:
        if state["lines"]:
            sections.append({"heading": state["heading"], "text": "\n".join(state["lines"])})
        state["lines"] = []

    block_cache: dict[int, Optional[Tag]] = {}

    def nearest_block(tag: Tag) -> Optional[Tag]:
        key = id(tag)
        if key in block_cache:
            return block_cache[key]
        node = tag
        while node is not None and not (isinstance(node, Tag) and node.name in BLOCK_TAGS):
            node = node.parent
        block_cache[key] = node
        return node

    current_block: Optional[Tag] = None
    buffer: list[str] = []

    def flush_line() -> None:
        nonlocal buffer
        text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        buffer = []
        if not text or current_block is None:
            return
        if current_block.name in heading_names:
            flush_section()
            state["heading"] = text
        elif current_block.find_parent(HEADING_TAGS) is None:
            state["lines"].append(text)

    root = soup.body or soup
    for node in root.descendants:
        if isinstance(node, Tag):
            if node.name == "br":
                flush_line()
                current_block = None
            continue
        if isinstance(node, Comment) or not isinstance(node, NavigableString):
            continue
        text = str(node).strip()
        if not text:
            continue
        block = nearest_block(node.parent)
        if block is not current_block:
            flush_line()
            current_block = block
        buffer.append(text)
    flush_line()
    flush_section()
    return sections


def extract_json_ld(soup: BeautifulSoup) -> list[dict]:
    import json

    blocks = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError) as exc:
            logger.warning("Malformed JSON-LD block skipped: %s", exc)
            continue
        if isinstance(parsed, list):
            blocks.extend(p for p in parsed if isinstance(p, dict))
        elif isinstance(parsed, dict):
            blocks.append(parsed)
    return blocks


def extract_meta_tags(soup: BeautifulSoup) -> dict[str, str]:
    meta: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name")
        value = tag.get("content")
        if key and value:
            meta[key] = value
    return meta


def needs_js_fallback(soup: BeautifulSoup, raw_text: str) -> bool:
    """Heuristic: does this response look like it needs JS rendering to
    show real content? True if there's no title, no heading, and very
    little visible text.
    """
    has_title = bool(soup.title and soup.title.get_text(strip=True))
    has_heading = bool(soup.find(HEADING_TAGS))
    return not has_title and not has_heading and len(raw_text) < 500


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_text_file(path: Path) -> Optional[str]:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None
