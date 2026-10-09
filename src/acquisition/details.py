"""Brand-agnostic helpers for reading structured product data (prices,
modes, ratings) out of the data each site embeds in its HTML.

Everything here is a literal read of what the page states; nothing is
estimated. Missing values stay None and callers record why in
``ProductDetails.notes``.
"""
from __future__ import annotations

import html as html_lib
import json
import re
from typing import Any, Optional

from .models import Mode

_DECODER = json.JSONDecoder()

# Price tolerance (currency units) for cross-checks between two sources.
PRICE_TOLERANCE = 0.011


def decode_json_at(text: str, start: int) -> Optional[Any]:
    """Decode the JSON value that begins exactly at ``text[start]``."""
    try:
        value, _ = _DECODER.raw_decode(text, start)
    except ValueError:
        return None
    return value


def unescape_embedded_json(html: str) -> str:
    """Philips ships its product data inside JS string chunks, so every JSON
    quote is escaped as \\". Removing that one escape level makes the
    objects decodable with ``decode_json_at``.
    """
    return html.replace('\\"', '"')


def js_json_parse_payload(html: str, var_name: str) -> Optional[dict]:
    """Oral-B (BigCommerce) ships ``var <name> = JSON.parse("<js string>")``.
    The argument is a JS string literal holding JSON, so it is decoded twice.
    """
    m = re.search(r"var " + re.escape(var_name) + r' = JSON\.parse\("((?:[^"\\]|\\.)*)"\)', html)
    if not m:
        return None
    try:
        literal = m.group(1).replace("\\'", "'")
        payload = json.loads(json.loads('"' + literal + '"'))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def compute_discount(original: Optional[float], sale: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Return (discount_amount, discount_pct) from original and sale price.
    (0.0, 0.0) when there is no markdown; (None, None) if either is unknown.
    """
    if original is None or sale is None or original <= 0:
        return None, None
    amount = round(original - sale, 2)
    if amount <= 0:
        return 0.0, 0.0
    return amount, round(amount / original * 100, 1)


def html_to_lines(fragment: str) -> list[str]:
    """Flatten an HTML fragment to text lines, one per block element."""
    # Source newlines are just soft wraps in HTML; only tags break lines.
    s = re.sub(r"\s+", " ", fragment)
    s = re.sub(r"(?i)<\s*(br|/p|/li|/div|/h\d|/tr)\b[^>]*>", "\n", s)
    s = re.sub(r"(?i)<\s*(li|p|div|h\d|tr)\b[^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html_lib.unescape(s).replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in s.split("\n")]
    return [ln for ln in lines if ln]


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_count(token: str) -> Optional[int]:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


# "7 Smart Brushing Modes", "4 CLEANING MODES", "5 Smart Modes", "3 different cleaning modes"
_MODE_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:[a-z/\-]+\s+){0,3}?modes?\b",
    re.I,
)
_MODE_LIST_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:[a-z/\-]+\s+){0,3}?modes?\b"
    r"[^:\n—–]{0,60}?(?::|—|–|\bincluding\b|\bare\b)\s*([^\n]+)",
    re.I,
)
_DESC_SPLIT_RE = re.compile(r"\s+(for|to)\s+", re.I)


def find_stated_mode_count(texts: list[str]) -> Optional[int]:
    """First 'N ... modes' count stated in any of the given strings."""
    for text in texts:
        for m in _MODE_COUNT_RE.finditer(text):
            n = parse_count(m.group(1))
            if n:
                return n
    return None


def _clean_mode_name(name: str) -> str:
    name = re.sub(r"^(?:and|including)\s+", "", name.strip(), flags=re.I)
    return name.strip(" .;:-—–")


def _title_if_lower(name: str) -> str:
    """'super-sensitive' / 'daily clean' -> 'Super-Sensitive' / 'Daily Clean'
    (names the page writes in lower case); mixed-case names are kept."""
    return name.title() if name.islower() else name


def parse_mode_list(line: str) -> list[Mode]:
    """Parse 'Daily Clean for everyday teeth, Whiten to make teeth brighter,
    and Tongue Clean' style enumerations into Mode objects."""
    # The list ends at the first sentence break or closing em dash.
    body = re.split(r"(?<=[a-z\)])\.(?:\s|$)|—|–|\s{2,}", line, maxsplit=1)[0]
    parts = [p for p in re.split(r",\s*|\s+and\s+(?=[A-Z])", body) if p.strip()]
    modes: list[Mode] = []
    for part in parts:
        part = _clean_mode_name(part)
        if not part:
            continue
        halves = _DESC_SPLIT_RE.split(part, maxsplit=1)
        if len(halves) == 3:
            name, desc = _clean_mode_name(halves[0]), halves[1] + " " + halves[2]
            modes.append(Mode(name=_title_if_lower(name), description=desc.strip(" .")))
        else:
            modes.append(Mode(name=_title_if_lower(part)))
    return modes


def extract_modes_from_lines(lines: list[str]) -> tuple[Optional[int], list[Mode]]:
    """Find an 'N ... modes: A, B, C' enumeration in page text lines.

    Returns (stated_count, modes). The line whose parsed list length equals
    the stated count wins; otherwise the longest parsed list is returned so
    the caller can flag the mismatch.
    """
    best: list[Mode] = []
    stated = find_stated_mode_count(lines)
    for line in lines:
        m = _MODE_LIST_RE.search(line)
        if not m:
            continue
        count = parse_count(m.group(1))
        modes = parse_mode_list(m.group(2))
        if count and len(modes) == count:
            return count, modes
        if len(modes) > len(best):
            best = modes
    return stated, best



# ---------------------------------------------------------------------------
# Oral-B embedded product copy (mainProduct.warranty): description bullets
# plus accordion tabs (What's in the box, Make the most of every session,
# Featured in, Warranty). The tab bodies are rendered client-side, so they
# are not in the static DOM text; this is where the shopper-visible copy is.
# ---------------------------------------------------------------------------

_TAB_HEADINGS = {
    "tab-whats-included": "What's in the box",
    "tab-make-the-most-every-session": "Make the most of every session",
    "tab-featured-in": "Featured in",
    "tab-warranty": "Warranty information",
}

_FOOTNOTE_LINE_RE = re.compile(r"^(?P<marker>[*\u2020\u2021\u00a7]+)\s*(?P<text>\S.*)$")
_INLINE_FOOTNOTE_RE = re.compile(r"\s(?P<marker>\*+)\s*(?P<text>(?:vs\.?|versus|based on)\b[^*]*)$", re.I)


def parse_product_copy(warranty_html: str) -> dict:
    """Split the embedded copy into ``intro`` (description bullets, each as
    (line, bold_lead_in)) and ``tabs`` {heading: [lines]}."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup("<div id='root'>" + (warranty_html or "") + "</div>", "lxml")
    root = soup.find(id="root")
    tabs: dict[str, list[str]] = {}
    for tab in root.select("div.tab-content"):
        heading = _TAB_HEADINGS.get(tab.get("id") or "", (tab.get("id") or "tab").replace("tab-", "").replace("-", " ").capitalize())
        tabs[heading] = html_to_lines(str(tab))
        tab.decompose()
    intro: list[tuple[str, str]] = []
    for block in root.find_all(["li", "p"]):
        if block.find_parent(["li", "p"]) is not None and block.name == "p" and block.find_parent("li"):
            continue
        for line in html_to_lines(str(block)):
            lead = block.find(["strong", "b"])
            strong = lead.get_text(" ", strip=True) if lead else ""
            intro.append((line, strong))
    # de-duplicate while keeping order (nested <li><p> repeats the line)
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for line, strong in intro:
        if line not in seen:
            seen.add(line)
            unique.append((line, strong))
    return {"intro": unique, "tabs": tabs}


_DIMS_RE = re.compile(r"\s*(?P<dims>Height:.*?)(?=\s*Weight:|$)", re.I)
_WEIGHT_RE = re.compile(r"\s*(?P<weight>Weight:.*)$", re.I)
_BOX_TRAILING_NOTE_RE = re.compile(r"\s*\((?:we recommend|recommended)[^)]*\)\s*$", re.I)


def split_box_item(line: str) -> tuple[str, Optional[str], Optional[str]]:
    """'Toothbrush. Height: ...; Weight: ...' -> (item, dimensions, weight);
    also strips a trailing '(We recommend ...)' note."""
    weight = dims = None
    m = _WEIGHT_RE.search(line)
    if m:
        weight = m.group("weight").strip()
        line = line[: m.start()]
    m = _DIMS_RE.search(line)
    if m:
        dims = m.group("dims").strip().rstrip(";.")
        line = line[: m.start()]
    line = _BOX_TRAILING_NOTE_RE.sub("", line).strip().rstrip(".").strip()
    return line, dims, weight


def extract_footnotes(lines: list[str], section: Optional[str] = None) -> list[dict]:
    """Footnote/qualifier lines: either a whole line starting with a marker
    run ('*vs. a regular manual toothbrush') or a trailing inline footnote
    ('... *vs a manual brush.')."""
    found: list[dict] = []
    for line in lines:
        m = _FOOTNOTE_LINE_RE.match(line)
        if m:
            found.append({"marker": m.group("marker"), "text": m.group("text").strip(), "section": section})
            continue
        m = _INLINE_FOOTNOTE_RE.search(line)
        if m and "*" in line[: m.start()]:
            found.append({"marker": m.group("marker"), "text": m.group("text").strip(), "section": section})
    return found



# ---------------------------------------------------------------------------
# Bundles: count HANDLES only (a handle + 2 brush heads is one product)
# ---------------------------------------------------------------------------

_HANDLE_WORD = re.compile(r"\b(?:handles?|toothbrush(?:es)?)\b", re.I)
_NOT_A_HANDLE = re.compile(
    r"brush[- ]?heads?|\bheads?\b|refill|travel|\bcase\b|pouch|charger|charging|\bstand\b|\bbase\b|"
    r"\bcord\b|batter(?:y|ies)|guide|manual|brochure|toothpaste|floss",
    re.I,
)
_BUNDLE_NAME = re.compile(
    r"\b(?:twin|dual|duo|double)[- ]?pack\b|\b(?:2|two)[- ]?pack\b|\bbundle\b|\bvalue pack\b", re.I
)


def _quantity(item: str) -> int:
    m = re.match(r"\s*\((\d+)\)", item) or re.match(r"\s*(\d+)\s*[x×]\s", item)
    if m:
        return int(m.group(1))
    m = re.search(r":\s*(\d+)\b", item)
    return int(m.group(1)) if m else 1


def count_handles(box_items: list[str]) -> int:
    total = 0
    for item in box_items:
        if _HANDLE_WORD.search(item) and not _NOT_A_HANDLE.search(item):
            total += _quantity(item)
    return total


def classify_bundle(product_name: str, box_items: list[str]) -> tuple[str, int, Optional[str]]:
    """(product_type, handle_count, reason). A product is a bundle when its
    name says Twin/Dual/Duo pack or bundle, or its box holds 2+ handles.
    Brush heads, cases and chargers are never counted."""
    handles = count_handles(box_items)
    if _BUNDLE_NAME.search(product_name or ""):
        return "bundle", handles, f"name says bundle/multi-pack ({handles} handle(s) in the box)"
    if handles >= 2:
        return "bundle", handles, f"{handles} handles in the box"
    return "single", handles, None
