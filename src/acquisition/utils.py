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
    return soup


def extract_visible_text(soup: BeautifulSoup) -> str:
    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def extract_sections(soup: BeautifulSoup) -> list[dict]:
    """Walk headings and following text-bearing tags in document order,
    grouping text under the most recent heading. Not based on nth-child
    position -- based on tag semantics only.
    """
    sections: list[dict] = []
    current_heading = "Page content"
    current_texts: list[str] = []

    root = soup.body or soup
    for tag in root.find_all(HEADING_TAGS + TEXT_TAGS):
        if tag.name in HEADING_TAGS:
            if current_texts:
                sections.append({"heading": current_heading, "text": "\n".join(current_texts)})
            heading_text = tag.get_text(" ", strip=True)
            current_heading = heading_text or current_heading
            current_texts = []
        else:
            # Skip text that is itself nested inside a heading tag.
            if tag.find_parent(HEADING_TAGS):
                continue
            txt = tag.get_text(" ", strip=True)
            if txt:
                current_texts.append(txt)

    if current_texts:
        sections.append({"heading": current_heading, "text": "\n".join(current_texts)})
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
