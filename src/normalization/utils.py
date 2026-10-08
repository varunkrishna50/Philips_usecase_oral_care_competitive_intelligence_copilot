"""Deterministic, brand-agnostic attribute extraction.

Every function here operates ONLY over usable_sections() output (never
record["raw_text"] directly) so that reviews, cross-sell, and
comparison-FAQ content structurally cannot become evidence -- see
filters.py. Price is the one exception: it never scans free text at all,
only the two structured channels (JSON-LD offers.price, and Oral-B's
price_text), per the Step 2 spec's price-conservatism rule.

Nothing here infers a value that isn't explicitly supported by matched
text. Every regex is intentionally narrow (see the docstring on each
function for the concrete false-positive it is designed to avoid).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .evidence import EvidenceStore
from .models import AttributeValue, NormalizationWarning, not_mentioned

UsableSection = tuple[int, dict]  # (original_index, {"heading": ..., "text": ...})


# ---------------------------------------------------------------------------
# Snippet helpers
# ---------------------------------------------------------------------------

def combined_section_text(section: dict) -> str:
    return section["heading"] + "\n" + section["text"]


def searchable_text(section: dict, product_name: str) -> str:
    """Like combined_section_text(), except the heading is EXCLUDED when
    it's not a genuine assertion of fact:

    - An FAQ heading phrased as a question ("Does the Series 10 charger
      display... a brushing timer?") is not evidence of its own answer --
      the real answer lives in section["text"], and matching the keyword
      inside the question itself would wrongly treat the question as a
      positive statement.
    - A heading that's just the product's own name/title (Philips's and
      Oral-B's H1 is stored as its own section heading) is an identifier,
      not a claim -- e.g. "Philips One by Sonicare Battery Toothbrush"
      contains the word "battery" only because that's the product's own
      name, not because the page is stating a battery-life fact.

    A genuine declarative marketing heading ("Pressure Sensor helps
    protect your teeth and gums") is still included, since that line
    really is an assertion the extractors should be able to use.
    """
    heading = section["heading"].strip()
    if heading.endswith("?") or heading == product_name.strip():
        return section["text"]
    return heading + "\n" + section["text"]


def sentence_containing(text: str, start: int, end: int, max_len: int = 240) -> str:
    """A short, verbatim snippet of `text` covering [start:end), widened to
    nearby sentence boundaries. Always a true substring of `text`."""
    left_nl = text.rfind("\n", 0, start)
    left_period = text.rfind(". ", 0, start)
    left_bound = max(left_nl, left_period, -1) + 1
    right_nl = text.find("\n", end)
    right_period = text.find(". ", end)
    candidates = [c for c in (right_nl, right_period) if c != -1]
    right_bound = min(candidates) + 1 if candidates else len(text)
    snippet = text[left_bound:right_bound].strip()
    if not snippet or len(snippet) > max_len:
        snippet = text[max(0, start - 60):end + 60].strip()
    return snippet


def _find_matches(
    sections: list[UsableSection],
    pattern: re.Pattern,
    product_name: str = "",
) -> list[dict]:
    """Returns one dict per regex match, across all usable sections, in
    page order: {raw_value, evidence_text, section_heading, section_index,
    combined_text}. See searchable_text() for why FAQ-question and
    product-title headings are excluded from the search.
    """
    results = []
    for idx, section in sections:
        combined = searchable_text(section, product_name)
        for m in pattern.finditer(combined):
            results.append(
                {
                    "match": m,
                    "raw_value": m.group(0).strip(),
                    "evidence_text": sentence_containing(combined, m.start(), m.end()),
                    "section_heading": section["heading"],
                    "section_index": idx,
                    "combined_text": combined,
                }
            )
    return results


# ---------------------------------------------------------------------------
# Generic boolean attribute (stated / stated_absent / not_mentioned)
# ---------------------------------------------------------------------------

def extract_boolean_attribute(
    sections: list[UsableSection],
    product_id: str,
    attribute: str,
    record: dict,
    store: EvidenceStore,
    warnings: list[NormalizationWarning],
    positive_pattern: re.Pattern,
    negative_pattern: Optional[re.Pattern] = None,
) -> AttributeValue:
    product_name = record.get("product_name", "")
    positives = _find_matches(sections, positive_pattern, product_name)
    negatives = _find_matches(sections, negative_pattern, product_name) if negative_pattern else []

    # A negation like "does not have a timer" also contains the bare word
    # "timer", and a question-framed heading ("Does the Series 2 have a
    # pressure sensor?") restates the keyword right next to its own
    # answer. Neither is an independent, conflicting positive statement --
    # a negation anywhere in a section is that section's answer, and
    # overrides any positive-sounding keyword elsewhere in the SAME
    # section (including its own heading). Only a positive match from a
    # genuinely different section still counts as competing evidence.
    negative_section_indices = {n["section_index"] for n in negatives}
    positives = [p for p in positives if p["section_index"] not in negative_section_indices]

    if positives and negatives:
        warnings.append(
            NormalizationWarning(
                product_id=product_id,
                code="conflicting_boolean_evidence",
                attribute=attribute,
                message=(
                    f"Both a positive ({positives[0]['raw_value']!r}) and a negative "
                    f"({negatives[0]['raw_value']!r}) statement were found for {attribute}."
                ),
            )
        )
        eids = []
        for m, val, status in [(positives[0], True, "ambiguous"), (negatives[0], False, "ambiguous")]:
            eids.append(
                store.add_from_section(
                    product_id=product_id,
                    attribute=attribute,
                    value=None,
                    status="ambiguous",
                    raw_value=m["raw_value"],
                    evidence_text=m["evidence_text"],
                    source_url=record["product_url"],
                    source_file=record["raw_html_path"],
                    section_heading=m["section_heading"],
                    section_index=m["section_index"],
                    section_combined_text=m["combined_text"],
                )
            )
        return AttributeValue(value=None, status="ambiguous", raw_value=None, evidence_ids=eids)

    if positives:
        m = positives[0]
        eid = store.add_from_section(
            product_id=product_id,
            attribute=attribute,
            value=True,
            status="stated",
            raw_value=m["raw_value"],
            evidence_text=m["evidence_text"],
            source_url=record["product_url"],
            source_file=record["raw_html_path"],
            section_heading=m["section_heading"],
            section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        return AttributeValue(value=True, status="stated", raw_value=m["raw_value"], evidence_ids=[eid])

    if negatives:
        m = negatives[0]
        eid = store.add_from_section(
            product_id=product_id,
            attribute=attribute,
            value=False,
            status="stated_absent",
            raw_value=m["raw_value"],
            evidence_text=m["evidence_text"],
            source_url=record["product_url"],
            source_file=record["raw_html_path"],
            section_heading=m["section_heading"],
            section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        return AttributeValue(value=False, status="stated_absent", raw_value=m["raw_value"], evidence_ids=[eid])

    return not_mentioned()


# ---------------------------------------------------------------------------
# Price -- structured channels only, never free text
# ---------------------------------------------------------------------------

def _coerce_price_number(raw: Any) -> Optional[float]:
    """JSON-LD price is a number on Philips pages but a numeric STRING on
    Oral-B pages (e.g. "169.99") -- accept either, but never guess at a
    value from something that isn't cleanly numeric."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if re.fullmatch(r"\d+(\.\d+)?", stripped):
            return float(stripped)
    return None


# schema.org/ItemAvailability values that mean "this is not currently a
# purchasable, active price" -- a listed price next to one of these is
# the catalog/last-known price, not a price a visitor can act on today.
_UNAVAILABLE_SCHEMA_STATES = {
    "outofstock", "soldout", "discontinued",
}


def _jsonld_product_offer(record: dict) -> tuple[Optional[float], Optional[str], Optional[str]]:
    json_ld = record.get("structured_data", {}).get("json_ld", [])
    for block in json_ld:
        if isinstance(block, dict) and block.get("@type") == "Product":
            offers = block.get("offers")
            if isinstance(offers, dict):
                price = _coerce_price_number(offers.get("price"))
                if price is not None:
                    currency = offers.get("priceCurrency")
                    availability = offers.get("availability")
                    return (
                        price,
                        currency if isinstance(currency, str) else None,
                        availability if isinstance(availability, str) else None,
                    )
    return None, None, None


def _price_text_value(record: dict) -> tuple[Optional[float], Optional[str]]:
    price_text = record.get("structured_data", {}).get("price_text")
    if not isinstance(price_text, str):
        return None, None
    m = re.search(r"[\d,]+(?:\.\d{1,2})?", price_text)
    if not m:
        return None, None
    try:
        return float(m.group(0).replace(",", "")), price_text
    except ValueError:
        return None, None


def _jsonld_price_snippet(html_text: str, value: float) -> str:
    """A verbatim snippet of the raw HTML containing the price, located by
    searching for how json.dumps would plausibly have rendered it. Falls
    back to the bare numeral if the exact '"price":X' form isn't found."""
    candidates = [f"{value:.2f}"]
    if value == int(value):
        candidates.append(str(int(value)))
    candidates.append(repr(value))
    for cand in candidates:
        for prefix in (f'"price":{cand}', f'"price": {cand}', f'"price":"{cand}"', f'"price": "{cand}"'):
            idx = html_text.find(prefix)
            if idx != -1:
                end = min(len(html_text), idx + len(prefix) + 40)
                return html_text[idx:end]
    for cand in candidates:
        idx = html_text.find(cand)
        if idx != -1:
            return html_text[idx:idx + len(cand)]
    return f"{value:.2f}"


def _jsonld_availability_snippet(html_text: str, availability: str) -> str:
    for prefix in (f'"availability":"{availability}"', f'"availability": "{availability}"'):
        idx = html_text.find(prefix)
        if idx != -1:
            return html_text[idx:idx + len(prefix)]
    return availability


_RRP_PRICE_ATTR = re.compile(r"data-product-rrp-price-without-tax[^>]*>\s*\$?([\d,]+\.\d{2})", re.I)


def _secondary_rrp_prices(html_text: str) -> list[float]:
    """Numeric values found in BigCommerce's RRP/original-price elements
    (data-product-rrp-price-without-tax) anywhere on the raw page -- see
    extract_price()'s caller for why these are surfaced as a warning
    rather than acted on automatically."""
    values = []
    for m in _RRP_PRICE_ATTR.finditer(html_text):
        try:
            values.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    return values


def extract_price(
    record: dict,
    product_id: str,
    store: EvidenceStore,
    warnings: list[NormalizationWarning],
) -> tuple[AttributeValue, AttributeValue, AttributeValue]:
    json_ld_price, json_ld_currency, json_ld_availability = _jsonld_product_offer(record)
    text_price, price_text_raw = _price_text_value(record)

    candidates = []
    if json_ld_price is not None:
        candidates.append(("json_ld", json_ld_price, json_ld_currency))
    if text_price is not None:
        candidates.append(("price_text", text_price, None))

    if not candidates:
        warnings.append(
            NormalizationWarning(
                product_id=product_id,
                code="no_price_signal",
                attribute="price",
                message="No structured price signal (JSON-LD offer or price_text) was captured for this product.",
            )
        )
        return not_mentioned(), not_mentioned(), not_mentioned()

    html_path = Path(record["raw_html_path"])
    html_text = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
    distinct_values = {round(v, 2) for _, v, _ in candidates}

    def _evidence_for(source: str, value: float) -> tuple[str, str]:
        if source == "price_text":
            return price_text_raw, price_text_raw
        return str(value), _jsonld_price_snippet(html_text, value)

    if len(distinct_values) > 1:
        warnings.append(
            NormalizationWarning(
                product_id=product_id,
                code="price_conflict",
                attribute="price",
                message=f"Conflicting standalone-price signals found: {candidates}",
            )
        )
        price_eids = []
        for source, value, _cur in candidates:
            raw_value, evidence_text = _evidence_for(source, value)
            price_eids.append(
                store.add_from_structured(
                    product_id=product_id,
                    attribute="price",
                    value=value,
                    status="ambiguous",
                    raw_value=raw_value,
                    evidence_text=evidence_text,
                    source_url=record["product_url"],
                    source_file=record["raw_html_path"],
                    raw_text=record["raw_text"],
                    html_text=html_text,
                )
            )
        price_av = AttributeValue(value=None, status="ambiguous", raw_value=None, evidence_ids=price_eids)
        return price_av, not_mentioned(), AttributeValue(value=None, status="ambiguous", evidence_ids=[])

    # Agreement (or a single source): priority is json_ld > price_text, per
    # the documented source-priority order -- but since they agree here,
    # this only matters for which raw_value/evidence_text gets used.
    source, value, currency = candidates[0]
    raw_value, evidence_text = _evidence_for(source, value)

    # Oral-B pages can carry a second "RRP"/original-price element (e.g.
    # BigCommerce's data-product-rrp-price-without-tax) alongside the
    # primary price -- sometimes genuinely a sale-price pairing, sometimes
    # tied to a bundle/subscription card elsewhere on the page (confirmed
    # by inspection: one product's RRP block showed the *other* purchase
    # option's price, not a "was/now" pair for this exact offer). Rather
    # than guess which case applies, surface it as a warning and leave
    # price/price_type untouched -- a human can decide with the raw HTML.
    secondary_prices = sorted(
        {v for v in _secondary_rrp_prices(html_text) if round(v, 2) != round(value, 2)}
    )
    if secondary_prices:
        warnings.append(
            NormalizationWarning(
                product_id=product_id,
                code="secondary_price_signal_present",
                attribute="price",
                message=(
                    f"Page also contains other price value(s) {secondary_prices} distinct from the "
                    f"extracted standalone price ${value:.2f} (e.g. an RRP/original-price element, a "
                    "bundle option, or a subscription card). Not used -- review raw HTML if a sale/original "
                    "price distinction matters here."
                ),
            )
        )

    price_eid = store.add_from_structured(
        product_id=product_id, attribute="price", value=value, status="stated",
        raw_value=raw_value, evidence_text=evidence_text,
        source_url=record["product_url"], source_file=record["raw_html_path"],
        raw_text=record["raw_text"], html_text=html_text,
    )
    price_av = AttributeValue(value=value, status="stated", raw_value=raw_value, evidence_ids=[price_eid])

    if currency:
        currency_eid = store.add_from_structured(
            product_id=product_id, attribute="currency", value=currency, status="stated",
            raw_value=currency, evidence_text=evidence_text,
            source_url=record["product_url"], source_file=record["raw_html_path"],
            raw_text=record["raw_text"], html_text=html_text,
        )
        currency_av = AttributeValue(value=currency, status="stated", raw_value=currency, evidence_ids=[currency_eid])
    else:
        currency_av = not_mentioned()

    # A price next to an explicit "OutOfStock"/"SoldOut"/"Discontinued"
    # availability is the catalog's last-known price, not one a visitor
    # can currently act on -- price_type must say so rather than silently
    # presenting it as an active, purchasable "listed" price.
    availability_state = None
    if json_ld_availability:
        availability_state = json_ld_availability.rsplit("/", 1)[-1].strip().lower()
    is_unavailable = availability_state in _UNAVAILABLE_SCHEMA_STATES

    if is_unavailable:
        price_type_value = "unavailable"
        price_type_raw = json_ld_availability
        price_type_evidence_text = _jsonld_availability_snippet(html_text, json_ld_availability)
        warnings.append(
            NormalizationWarning(
                product_id=product_id,
                code="price_unavailable",
                attribute="price_type",
                message=(
                    f"Product offer availability is {json_ld_availability!r}; the ${value:.2f} price "
                    "is the catalog's last-known price, not a currently purchasable one."
                ),
            )
        )
    else:
        price_type_value = "listed"
        price_type_raw = raw_value
        price_type_evidence_text = evidence_text

    price_type_eid = store.add_from_structured(
        product_id=product_id, attribute="price_type", value=price_type_value, status="stated",
        raw_value=price_type_raw, evidence_text=price_type_evidence_text,
        source_url=record["product_url"], source_file=record["raw_html_path"],
        raw_text=record["raw_text"], html_text=html_text,
    )
    price_type_av = AttributeValue(value=price_type_value, status="stated", raw_value=price_type_raw, evidence_ids=[price_type_eid])

    return price_av, currency_av, price_type_av


# ---------------------------------------------------------------------------
# Pressure sensing
# ---------------------------------------------------------------------------

PRESSURE_POSITIVE = re.compile(
    r"pressure sensor|pressure control|pressure sensing|too (?:much|hard) pressure|"
    r"brushing too (?:hard|lightly)|applying too much pressure|"
    r"alerts? you (?:when|if) you'?re brushing too hard",
    re.I,
)
PRESSURE_NEGATIVE = re.compile(
    r"does not have (?:a )?pressure sensor|without (?:a )?pressure sensor|no pressure sensor",
    re.I,
)


def extract_pressure_sensing(sections, product_id, record, store, warnings) -> AttributeValue:
    return extract_boolean_attribute(
        sections, product_id, "pressure_sensing", record, store, warnings,
        PRESSURE_POSITIVE, PRESSURE_NEGATIVE,
    )


# ---------------------------------------------------------------------------
# App connectivity vs Bluetooth -- kept strictly separate, see Phase 1
# finding: several products state Bluetooth with zero "app" mention.
# ---------------------------------------------------------------------------

APP_POSITIVE = re.compile(r"\bapp\b", re.I)
APP_NEGATIVE = re.compile(
    r"does not (?:have|support|require) (?:the )?app|no app (?:connectivity|required|needed)|without (?:the |an )?app\b",
    re.I,
)
BLUETOOTH_POSITIVE = re.compile(r"bluetooth", re.I)
BLUETOOTH_NEGATIVE = re.compile(r"does not have bluetooth|no bluetooth|without bluetooth", re.I)


def extract_app_connectivity(sections, product_id, record, store, warnings) -> AttributeValue:
    return extract_boolean_attribute(
        sections, product_id, "app_connectivity", record, store, warnings,
        APP_POSITIVE, APP_NEGATIVE,
    )


def extract_bluetooth_connectivity(sections, product_id, record, store, warnings) -> AttributeValue:
    return extract_boolean_attribute(
        sections, product_id, "bluetooth_connectivity", record, store, warnings,
        BLUETOOTH_POSITIVE, BLUETOOTH_NEGATIVE,
    )


# ---------------------------------------------------------------------------
# Timer (bool + free-text description)
# ---------------------------------------------------------------------------

TIMER_POSITIVE = re.compile(
    r"\btimer\b|smartimer|quadpacer|quadrant (?:interval|pacer)|sextant interval|"
    r"\b2[\s-]minute\b|\b30[\s-]second\b",
    re.I,
)
TIMER_NEGATIVE = re.compile(r"does not have a timer|no (?:built-in )?timer|without a timer", re.I)


def extract_timer(sections, product_id, record, store, warnings) -> tuple[AttributeValue, AttributeValue]:
    timer_av = extract_boolean_attribute(
        sections, product_id, "timer", record, store, warnings, TIMER_POSITIVE, TIMER_NEGATIVE,
    )
    if timer_av.status != "stated":
        # stated_absent / not_mentioned / ambiguous all leave the separate
        # free-text description field unpopulated -- it has no independent
        # meaning once "is there a timer at all" isn't a confirmed yes.
        return timer_av, not_mentioned()

    ev = store.get(timer_av.evidence_ids[0])
    desc_text = ev.evidence_text
    section = record["sections"][ev.source_section_index]
    desc_eid = store.add_from_section(
        product_id=product_id, attribute="timer_description", value=desc_text, status="stated",
        raw_value=desc_text, evidence_text=desc_text,
        source_url=record["product_url"], source_file=record["raw_html_path"],
        section_heading=section["heading"], section_index=ev.source_section_index,
        section_combined_text=combined_section_text(section),
    )
    desc_av = AttributeValue(value=desc_text, status="stated", raw_value=desc_text, evidence_ids=[desc_eid])
    return timer_av, desc_av


# ---------------------------------------------------------------------------
# Battery life
# ---------------------------------------------------------------------------

BATTERY_NUM_BEFORE = re.compile(r"\b(\d+)[\s-]*(days?|weeks?)\b[^.\n]{0,30}battery", re.I)
BATTERY_NUM_AFTER = re.compile(r"battery[^.\n]{0,40}?\b(\d+)[\s-]*(days?|weeks?)\b", re.I)
# "1 charge lasts 4 weeks" (e.g. under a "Long-lasting battery:" label) --
# a direct runtime statement that doesn't always sit within 40 chars of
# the literal word "battery" once a label/newline separates them.
BATTERY_CHARGE_LASTS = re.compile(r"charge lasts\s+(\d+)[\s-]*(days?|weeks?)\b", re.I)
BATTERY_MENTION = re.compile(r"\bbattery\b", re.I)
# Statements ABOUT a level indicator/recharge-reminder feature, or about
# the product being battery-POWERED (a product-type/name descriptor), are
# not runtime/duration claims and must not populate battery_life at all --
# not even as "stated but unnormalizable". E.g. "Battery Toothbrush" (a
# product name), "tracks your battery level", "low-charge indicator".
BATTERY_INDICATOR_NOISE = re.compile(
    r"battery\s+(?:toothbrush|level|status|indicator)|"
    r"tracks?\s+(?:your\s+)?battery|low-charge indicator|"
    r"when it.s time to recharge|recharge when",
    re.I,
)


def _days_from_unit(n: int, unit: str) -> int:
    return n * 7 if unit.lower().startswith("week") else n


def extract_battery_life(sections, product_id, record, store, warnings) -> tuple[AttributeValue, AttributeValue]:
    product_name = record.get("product_name", "")
    numeric_matches = []
    for pattern in (BATTERY_NUM_BEFORE, BATTERY_NUM_AFTER, BATTERY_CHARGE_LASTS):
        for idx, section in sections:
            combined = searchable_text(section, product_name)
            for m in pattern.finditer(combined):
                days = _days_from_unit(int(m.group(1)), m.group(2))
                numeric_matches.append(
                    {
                        "days": days,
                        "raw_value": m.group(0).strip(),
                        "evidence_text": sentence_containing(combined, m.start(), m.end()),
                        "section_heading": section["heading"],
                        "section_index": idx,
                        "combined_text": combined,
                    }
                )

    if numeric_matches:
        distinct_days = {m["days"] for m in numeric_matches}
        if len(distinct_days) > 1:
            warnings.append(
                NormalizationWarning(
                    product_id=product_id, code="battery_life_conflict", attribute="battery_life_days",
                    message=f"Conflicting battery-life durations found: {sorted(distinct_days)} days.",
                )
            )
            eids_days, eids_raw = [], []
            for m in numeric_matches:
                eids_days.append(store.add_from_section(
                    product_id=product_id, attribute="battery_life_days", value=None, status="ambiguous",
                    raw_value=m["raw_value"], evidence_text=m["evidence_text"],
                    source_url=record["product_url"], source_file=record["raw_html_path"],
                    section_heading=m["section_heading"], section_index=m["section_index"],
                    section_combined_text=m["combined_text"],
                ))
                eids_raw.append(store.add_from_section(
                    product_id=product_id, attribute="battery_life_raw", value=None, status="ambiguous",
                    raw_value=m["raw_value"], evidence_text=m["evidence_text"],
                    source_url=record["product_url"], source_file=record["raw_html_path"],
                    section_heading=m["section_heading"], section_index=m["section_index"],
                    section_combined_text=m["combined_text"],
                ))
            return (
                AttributeValue(value=None, status="ambiguous", evidence_ids=eids_days),
                AttributeValue(value=None, status="ambiguous", evidence_ids=eids_raw),
            )

        m = numeric_matches[0]
        days_eid = store.add_from_section(
            product_id=product_id, attribute="battery_life_days", value=m["days"], status="stated",
            raw_value=m["raw_value"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        raw_eid = store.add_from_section(
            product_id=product_id, attribute="battery_life_raw", value=m["raw_value"], status="stated",
            raw_value=m["raw_value"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        return (
            AttributeValue(value=m["days"], status="stated", raw_value=m["raw_value"], evidence_ids=[days_eid]),
            AttributeValue(value=m["raw_value"], status="stated", raw_value=m["raw_value"], evidence_ids=[raw_eid]),
        )

    # Battery discussed qualitatively but no parseable number -> stated,
    # value=null, per the Step 2 "stated but unnormalizable" rule. Matches
    # that are really about a level-indicator feature, a recharge
    # reminder, or the product's own "Battery Toothbrush" name/type are
    # excluded entirely (not even unnormalizable "stated") -- those are a
    # different concept from battery life/runtime.
    mention_matches = [
        m for m in _find_matches(sections, BATTERY_MENTION, product_name)
        if not BATTERY_INDICATOR_NOISE.search(m["evidence_text"])
    ]
    if mention_matches:
        m = mention_matches[0]
        warnings.append(
            NormalizationWarning(
                product_id=product_id, code="unnormalizable_value", attribute="battery_life_days",
                message="Battery life is explicitly mentioned but no numeric duration can be safely normalized.",
                section_heading=m["section_heading"],
            )
        )
        raw_eid = store.add_from_section(
            product_id=product_id, attribute="battery_life_raw", value=None, status="stated",
            raw_value=m["evidence_text"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        days_eid = store.add_from_section(
            product_id=product_id, attribute="battery_life_days", value=None, status="stated",
            raw_value=m["evidence_text"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        return (
            AttributeValue(value=None, status="stated", raw_value=m["evidence_text"], evidence_ids=[days_eid]),
            AttributeValue(value=m["evidence_text"], status="stated", raw_value=m["evidence_text"], evidence_ids=[raw_eid]),
        )

    return not_mentioned(), not_mentioned()


# ---------------------------------------------------------------------------
# Charging method -- "magnetic charger" only, never confused with
# "magnetic drive motor" (a brushing-motor spec, not a charging method).
# ---------------------------------------------------------------------------

CHARGING_MAGNETIC = re.compile(r"magnetic\s+(?:fast\s+)?charg\w*", re.I)
# More specific alternatives must come first: regex alternation takes the
# first alternative that matches at a given position, so a bare
# "charg(?:e|ing|er|ers)" listed first would win over "charging station"
# and truncate the match to just "Charging".
CHARGING_MENTION = re.compile(
    r"charging (?:station|stand|case|travel case)|smart charger|\bcharg(?:e|ing|er|ers)\b",
    re.I,
)


def extract_charging_method(sections, product_id, record, store, warnings) -> tuple[AttributeValue, AttributeValue]:
    product_name = record.get("product_name", "")
    magnetic_matches = _find_matches(sections, CHARGING_MAGNETIC, product_name)
    if magnetic_matches:
        m = magnetic_matches[0]
        method_eid = store.add_from_section(
            product_id=product_id, attribute="charging_method", value="magnetic", status="stated",
            raw_value=m["raw_value"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        raw_eid = store.add_from_section(
            product_id=product_id, attribute="charging_method_raw", value=m["raw_value"], status="stated",
            raw_value=m["raw_value"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        return (
            AttributeValue(value="magnetic", status="stated", raw_value=m["raw_value"], evidence_ids=[method_eid]),
            AttributeValue(value=m["raw_value"], status="stated", raw_value=m["raw_value"], evidence_ids=[raw_eid]),
        )

    mention_matches = _find_matches(sections, CHARGING_MENTION, product_name)
    if mention_matches:
        m = mention_matches[0]
        warnings.append(
            NormalizationWarning(
                product_id=product_id, code="unnormalizable_value", attribute="charging_method",
                message=f"Charging is mentioned ({m['raw_value']!r}) but cannot be safely mapped to a controlled charging-method category.",
                section_heading=m["section_heading"],
            )
        )
        method_eid = store.add_from_section(
            product_id=product_id, attribute="charging_method", value=None, status="stated",
            raw_value=m["raw_value"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        raw_eid = store.add_from_section(
            product_id=product_id, attribute="charging_method_raw", value=m["raw_value"], status="stated",
            raw_value=m["raw_value"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        return (
            AttributeValue(value=None, status="stated", raw_value=m["raw_value"], evidence_ids=[method_eid]),
            AttributeValue(value=m["raw_value"], status="stated", raw_value=m["raw_value"], evidence_ids=[raw_eid]),
        )

    return not_mentioned(), not_mentioned()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

# "brush" (not just "brushing") must be accepted: real Philips copy says
# "2 brush mode, 3 intensities" / "3 brush modes, 3 intensities".
MODE_COUNT_PATTERN = re.compile(r"\b(\d+)\s+(?:smart\s+)?(?:brushing\s+|brush\s+|cleaning\s+)?modes?\b", re.I)
# Intensity settings are a DIFFERENT concept from brushing modes and must
# never be converted into mode_count (e.g. Philips's "2 intensity
# settings for a more powerful or gentle clean" is not "2 modes") -- kept
# as its own pattern, deliberately never merged into count_matches below.
INTENSITY_COUNT_PATTERN = re.compile(r"\b(\d+)\s+intensity\s+settings?\b", re.I)
NAMED_MODES_PATTERN = re.compile(r"\bmodes?\s*(?:include|:|are)\s*([A-Z][\w' ]+(?:,\s*(?:and\s+)?[A-Z][\w' ]+){1,9})", re.I)


def extract_modes(sections, product_id, record, store, warnings) -> tuple[AttributeValue, AttributeValue]:
    product_name = record.get("product_name", "")
    count_matches = []
    for pattern in (MODE_COUNT_PATTERN,):
        for idx, section in sections:
            combined = searchable_text(section, product_name)
            for m in pattern.finditer(combined):
                count_matches.append(
                    {
                        "count": int(m.group(1)),
                        "raw_value": m.group(0).strip(),
                        "evidence_text": sentence_containing(combined, m.start(), m.end()),
                        "section_heading": section["heading"],
                        "section_index": idx,
                        "combined_text": combined,
                    }
                )

    mode_count_av = not_mentioned()
    if count_matches:
        distinct_counts = {m["count"] for m in count_matches}
        if len(distinct_counts) > 1:
            warnings.append(
                NormalizationWarning(
                    product_id=product_id, code="mode_count_conflict", attribute="mode_count",
                    message=f"Conflicting mode/intensity counts found: {sorted(distinct_counts)}.",
                )
            )
            eids = [
                store.add_from_section(
                    product_id=product_id, attribute="mode_count", value=None, status="ambiguous",
                    raw_value=m["raw_value"], evidence_text=m["evidence_text"],
                    source_url=record["product_url"], source_file=record["raw_html_path"],
                    section_heading=m["section_heading"], section_index=m["section_index"],
                    section_combined_text=m["combined_text"],
                )
                for m in count_matches
            ]
            mode_count_av = AttributeValue(value=None, status="ambiguous", evidence_ids=eids)
        else:
            m = count_matches[0]
            eid = store.add_from_section(
                product_id=product_id, attribute="mode_count", value=m["count"], status="stated",
                raw_value=m["raw_value"], evidence_text=m["evidence_text"],
                source_url=record["product_url"], source_file=record["raw_html_path"],
                section_heading=m["section_heading"], section_index=m["section_index"],
                section_combined_text=m["combined_text"],
            )
            mode_count_av = AttributeValue(value=m["count"], status="stated", raw_value=m["raw_value"], evidence_ids=[eid])

    modes_av = not_mentioned()
    named_matches = _find_matches(sections, NAMED_MODES_PATTERN, product_name)
    if named_matches:
        m = named_matches[0]
        names = [n.strip() for n in m["match"].group(1).split(",") if n.strip()]
        names = [re.sub(r"^and\s+", "", n, flags=re.I) for n in names]
        eid = store.add_from_section(
            product_id=product_id, attribute="modes", value=names, status="stated",
            raw_value=m["raw_value"], evidence_text=m["evidence_text"],
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=m["section_heading"], section_index=m["section_index"],
            section_combined_text=m["combined_text"],
        )
        modes_av = AttributeValue(value=names, status="stated", raw_value=m["raw_value"], evidence_ids=[eid])

    return mode_count_av, modes_av


# ---------------------------------------------------------------------------
# Included items -- only explicit "<N> <brush head/case/charger/...>"
# phrasing; conservative on purpose (Phase 1: Philips never has this in
# the captured static HTML at all; Oral-B has it on 9 of 13 products via
# a TextFeatureN-style attribute list).
# ---------------------------------------------------------------------------

INCLUDED_ITEM_LINE = re.compile(
    r"^\s*(\d+)[\s-]+(.*?(?:Brush Heads?|Refill Heads?|Replacement|Travel Case|Travel Pouch|"
    r"Charger|Handle|Charging (?:Case|Station)|\bBrush\b))\s*$",
    re.I,
)


def extract_included_items(sections, product_id, record, store, warnings) -> AttributeValue:
    items = []
    seen = set()
    eids = []
    for idx, section in sections:
        for line in section["text"].split("\n"):
            line_stripped = line.strip()
            m = INCLUDED_ITEM_LINE.match(line_stripped)
            if not m:
                continue
            if line_stripped in seen:
                continue
            seen.add(line_stripped)
            combined = combined_section_text(section)
            evidence_text = line_stripped
            if evidence_text not in combined:
                # extremely defensive: should not happen since the line
                # came directly from section["text"], which is part of
                # combined_section_text(); skip rather than risk bad evidence.
                continue
            eid = store.add_from_section(
                product_id=product_id, attribute="included_items", value=line_stripped, status="stated",
                raw_value=line_stripped, evidence_text=evidence_text,
                source_url=record["product_url"], source_file=record["raw_html_path"],
                section_heading=section["heading"], section_index=idx,
                section_combined_text=combined,
            )
            items.append(line_stripped)
            eids.append(eid)

    if not items:
        return not_mentioned()
    return AttributeValue(value=items, status="stated", raw_value=None, evidence_ids=eids)


# ---------------------------------------------------------------------------
# Claims -- verbatim comparative/marketing sentences
# ---------------------------------------------------------------------------

CLAIM_MARKER = re.compile(r"up to \d+%?x?|\d+%|(?<![\w@-])\d+x\b|\d+ ?times (?:more|better|less)", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# Image asset URLs/filenames (e.g. ".../Base-2x.png") leak into visible
# text alongside widget metadata on some pages; a "2x" retina-image
# density suffix must never be read as a "2x better" marketing claim.
_ASSET_LINE = re.compile(r"https?://|\.(?:png|jpe?g|svg|gif|webp)\b", re.I)
# Star-rating / review-count fragments (e.g. "4.3 | (3070) Reviews | 86%
# recommend this product") also contain a bare "%", which CLAIM_MARKER
# would otherwise mistake for a manufacturer performance claim -- this is
# aggregate customer-review statistics, not a claim about the product.
_REVIEW_STATS_LINE = re.compile(r"reviews\s*\|", re.I)

# Oral-B's page template repeats the same core description under several
# tab panels ("What's in the box", "Make the most of every session",
# "Featured in", "Warranty Information") and a "More Information" link,
# each rendered as its own short text node immediately followed -- with
# no separating punctuation -- by that same repeated sentence. Stripping
# a KNOWN exact label is safe (any suffix of a verbatim substring is
# still a verbatim substring); the suffix-based dedup below is a second,
# more general safety net for any glued prefix not in this list.
_KNOWN_GLUED_PREFIXES = (
    "More Information",
    "What's in the box",
    "What’s in the box",
    "Make the most of every session",
    "Featured in",
    "Warranty Information",
)
_GLUED_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in _KNOWN_GLUED_PREFIXES) + r")\s+"
)


def extract_claims(sections, product_id, record, store, warnings, max_claims: int = 6) -> AttributeValue:
    candidates: list[tuple[str, dict, int, str]] = []
    for idx, section in sections:
        combined = combined_section_text(section)
        for sentence in _SENTENCE_SPLIT.split(section["text"]):
            sentence = sentence.strip()
            sentence = _GLUED_PREFIX_RE.sub("", sentence).strip()
            if not sentence:
                continue
            if _ASSET_LINE.search(sentence) or _REVIEW_STATS_LINE.search(sentence):
                continue
            if not CLAIM_MARKER.search(sentence):
                continue
            if sentence not in combined:
                continue
            candidates.append((sentence, section, idx, combined))

    # Deduplicate: when two candidates differ only by one being a
    # (still-contaminated) longer version of the other -- i.e. one string
    # is a trailing substring of the other -- keep only the shorter,
    # cleaner one, regardless of which was found first.
    accepted: list[tuple[str, dict, int, str]] = []
    for cand in candidates:
        sentence = cand[0]
        skip = False
        for i, existing in enumerate(accepted):
            existing_sentence = existing[0]
            if sentence == existing_sentence:
                skip = True
                break
            if existing_sentence.endswith(sentence) and len(sentence) < len(existing_sentence):
                accepted[i] = cand  # this one is cleaner/shorter -- replace
                skip = True
                break
            if sentence.endswith(existing_sentence) and len(existing_sentence) <= len(sentence):
                skip = True  # existing is already the cleaner/shorter version
                break
        if not skip:
            accepted.append(cand)
        if len(accepted) >= max_claims:
            break

    claims: list[str] = []
    eids: list[str] = []
    for sentence, section, idx, combined in accepted[:max_claims]:
        eid = store.add_from_section(
            product_id=product_id, attribute="claims", value=sentence, status="stated",
            raw_value=sentence, evidence_text=sentence,
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=section["heading"], section_index=idx,
            section_combined_text=combined,
        )
        claims.append(sentence)
        eids.append(eid)

    if not claims:
        return not_mentioned()
    return AttributeValue(value=claims, status="stated", raw_value=None, evidence_ids=eids)


# ---------------------------------------------------------------------------
# Description -- first substantive, non-noise line of the product's own
# heading/intro section.
# ---------------------------------------------------------------------------

_NOISE_LINE_PATTERNS = [
    re.compile(r"^[A-Z0-9/]+$"),              # bare SKU codes, e.g. "HX9990/13"
    re.compile(r"^\d\.\d\s*\|"),              # "4.3 | (3070) Reviews..."
    re.compile(r"^\s*\|"),                    # leftover "| (3070) Reviews..." (missing its rating prefix)
    re.compile(r"reviews\s*\|", re.I),
    re.compile(r"recommend this product", re.I),
    re.compile(r"^available in$", re.I),
    re.compile(r"^★+$"),
    re.compile(r"^\(\d+\)"),
    re.compile(r"^\$"),
    re.compile(r"https?://|\.(?:png|jpe?g|svg|gif|webp)\b", re.I),  # stray asset URLs/filenames
    re.compile(r"-\d+x$", re.I),              # "<Color Name>-2x" retina-image swatch labels
    re.compile(r"^[\w .&'-]+:$"),             # bare "Label:" lines from widget/analytics metadata
    re.compile(r"^(?:fall|spring|summer|winter|holiday)?\s*savings?:", re.I),  # promo banners
    re.compile(r"\bsave \$\d", re.I),
    re.compile(r"limited quantit", re.I),
    re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$"),  # internal ids, e.g. "OralB_iO2_Electric_Toothbrush"
]


def _breadcrumb_names(record: dict) -> set[str]:
    """Breadcrumb trail labels (e.g. "Electric Toothbrushes", "iO Series")
    found verbatim in page text are category navigation, not a
    description of this specific product."""
    names: set[str] = set()
    for block in record.get("structured_data", {}).get("json_ld", []):
        if isinstance(block, dict) and block.get("@type") == "BreadcrumbList":
            for item in block.get("itemListElement", []):
                node = item.get("item") if isinstance(item, dict) else None
                name = node.get("name") if isinstance(node, dict) else None
                if isinstance(name, str):
                    names.add(name.strip().lower())
    return names


def _is_noise_line(line: str, breadcrumb_names: Optional[set[str]] = None) -> bool:
    if len(line) < 15:
        return True
    if breadcrumb_names and line.strip().lower() in breadcrumb_names:
        return True
    return any(p.search(line) for p in _NOISE_LINE_PATTERNS)


_DIMENSION_SPEC_LINE = re.compile(r"Height:.*Weight:|Weight:.*Height:", re.I)


def _looks_like_sentence(line: str, min_len: int = 50) -> bool:
    """A real description sentence, not a spec/feature bullet or an
    internal tracking code. A spec tag like "Smart Pressure Sensor" is
    short and made almost entirely of Title-Case words; a genuine
    sentence ("The iO Series 2 removes 100% more plaque even in
    hard-to-reach areas...") is longer and has several ordinary
    lowercase connector/verb words. Dimension/weight spec lines are
    excluded explicitly since they're long enough to otherwise pass.
    """
    if len(line) < min_len:
        return False
    if _DIMENSION_SPEC_LINE.search(line):
        return False
    if re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}", line, re.I):
        return False  # UUID-style internal identifier
    if len(line.split(" ")) < 6:
        return False  # a real sentence is space-separated, not hyphen-joined
    words = re.findall(r"[A-Za-z']+", line)
    lowercase_words = [w for w in words if w.islower() and len(w) >= 2]
    return len(lowercase_words) >= 4


def extract_description(sections, product_id, record, store, warnings) -> AttributeValue:
    breadcrumb_names = _breadcrumb_names(record)
    for idx, section in sections:
        for line in section["text"].split("\n"):
            line = line.strip()
            line = _GLUED_PREFIX_RE.sub("", line).strip()
            if not line or _is_noise_line(line, breadcrumb_names):
                continue
            if not _looks_like_sentence(line):
                continue
            combined = combined_section_text(section)
            if line not in combined:
                continue
            eid = store.add_from_section(
                product_id=product_id, attribute="description", value=line, status="stated",
                raw_value=line, evidence_text=line,
                source_url=record["product_url"], source_file=record["raw_html_path"],
                section_heading=section["heading"], section_index=idx,
                section_combined_text=combined,
            )
            return AttributeValue(value=line, status="stated", raw_value=line, evidence_ids=[eid])
    return not_mentioned()


# ---------------------------------------------------------------------------
# Specifications -- flat list of short verbatim feature/spec bullets.
# Brand-specific sourcing lives in philips.py/oralb.py; this just builds
# the final AttributeValue + evidence from a list of (line, section) pairs
# the caller has already identified as safe.
# ---------------------------------------------------------------------------

def build_specifications(
    product_id: str,
    record: dict,
    store: EvidenceStore,
    entries: list[tuple[str, dict, int]],  # (line, section, section_index)
) -> AttributeValue:
    specs: list[str] = []
    eids: list[str] = []
    seen = set()
    for line, section, idx in entries:
        line = line.strip()
        # Case-insensitive dedup: the two labeled-attribute systems
        # (TextFeatureN / Shop By Feature) sometimes restate the exact
        # same fact with different casing ("3 Smart Brushing Modes" vs
        # "3 SMART Brushing Modes").
        if not line or line.lower() in seen:
            continue
        combined = combined_section_text(section)
        if line not in combined:
            continue
        seen.add(line.lower())
        eid = store.add_from_section(
            product_id=product_id, attribute="specifications", value=line, status="stated",
            raw_value=line, evidence_text=line,
            source_url=record["product_url"], source_file=record["raw_html_path"],
            section_heading=section["heading"], section_index=idx,
            section_combined_text=combined,
        )
        specs.append(line)
        eids.append(eid)
    if not specs:
        return not_mentioned()
    return AttributeValue(value=specs, status="stated", raw_value=None, evidence_ids=eids)
