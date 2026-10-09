"""Comparison attributes read from Step 1's structured ``details`` and from
the product pages' spec data (pricing detail, colors, ratings, warranty,
battery type, charge time, brush speed, extra smart features, ...).

Same contract as the rest of Step 2: a value is only reported with an
evidence record whose text is a verbatim substring of the page (visible
text, raw HTML, the HTML with one JSON-escape level removed, or the
rendered rating-widget text). If no verbatim snippet can be found the
attribute is left not_mentioned -- never guessed. Booleans are only ever
True ("the page states this feature"); absence is not inferred.

Philips ships its spec table / feature list as embedded JSON, so for
Philips the free-text searches are scoped to those regions (never the
review text that sits elsewhere in the same HTML).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .evidence import EvidenceIntegrityError, EvidenceStore
from .models import AttributeValue, NormalizationWarning, not_mentioned

DETAIL_ATTRIBUTES = [
    "original_price",
    "discount_amount",
    "discount_pct",
    "price_after_discount",
    "promotion_text",
    "color_count",
    "colors",
    "rating_avg",
    "rating_count",
    "mode_details",
    "intensity_levels",
    "warranty_years",
    "battery_type",
    "charge_time_hours",
    "brush_movements_per_min",
    "display_type",
    "replacement_reminder",
    "coverage_tracking",
    "travel_case_included",
    "brush_heads_included",
    "money_back_guarantee_days",
    "plaque_removal_claim",
    "gum_health_claim",
    "dimensions",
    "weight",
    "disclaimers",
    "claims_with_disclaimers",
]

_DECODER = json.JSONDecoder()
_MAX_SNIPPET = 400


def _decode_with_end(text: str, start: int) -> tuple[Any, int] | None:
    try:
        return _DECODER.raw_decode(text, start)
    except ValueError:
        return None


class _Page:
    """Read-only view of one product page for evidence lookup."""

    def __init__(self, record: dict):
        self.record = record
        self.brand = record["brand"]
        self.details: dict = record.get("details") or {}
        path = Path(record["raw_html_path"])
        self.html = path.read_text(encoding="utf-8") if path.exists() else ""
        self.unescaped = self.html.replace('\\"', '"')
        self.raw_text = record["raw_text"]
        self.widget_text = record.get("structured_data", {}).get("rating_widget_text", "")
        self.chapters: dict[str, tuple[dict, str]] = {}
        self.scopes: list[str] = []
        if self.brand == "Philips":
            self._index_philips()
        else:
            self.scopes = [self.raw_text]

    # -- Philips spec JSON ---------------------------------------------------
    def _index_philips(self) -> None:
        for m in re.finditer(r'\{"csChapterCode"', self.unescaped):
            got = _decode_with_end(self.unescaped, m.start())
            if got is None:
                continue
            obj, end = got
            name = obj.get("csChapterName")
            if name and name not in self.chapters:
                self.chapters[name] = (obj, self.unescaped[m.start():end])
        self.scopes = [region for _, region in self.chapters.values()]
        idx = self.unescaped.find('"items":[{"featureLongDescription"')
        if idx >= 0:
            got = _decode_with_end(self.unescaped, idx + len('"items":'))
            if got is not None:
                self.scopes.append(self.unescaped[idx:got[1]])
                self.features_region = self.unescaped[idx:got[1]]

    def spec_item(self, name_re: str, chapter: Optional[str] = None) -> Optional[tuple[str, list[str], str]]:
        """(item name, values, verbatim item JSON) of the first matching spec item."""
        for cname, (obj, region) in self.chapters.items():
            if chapter and cname != chapter:
                continue
            for item in obj.get("csItem", []):
                iname = str(item.get("csItemName", ""))
                if re.fullmatch(name_re, iname, re.I):
                    values = [str(v.get("csValueName", "")).strip() for v in item.get("csValue", []) if v.get("csValueName")]
                    marker = f'"csItemName":"{iname}"'
                    at = region.find(marker)
                    start = region.rfind('{"csItemRank"', 0, at) if at >= 0 else -1
                    got = _decode_with_end(region, start) if start >= 0 else None
                    snippet = region[start:got[1]] if got else ""
                    return iname, values, snippet[:_MAX_SNIPPET]
        return None

    # -- free text -----------------------------------------------------------
    def search(self, pattern: str, flags: int = re.I) -> Optional[re.Match]:
        rx = re.compile(pattern, flags)
        for scope in self.scopes:
            m = rx.search(scope)
            if m:
                return m
        return None


def _add(
    page: _Page,
    store: EvidenceStore,
    warnings: list[NormalizationWarning],
    product_id: str,
    attribute: str,
    value: Any,
    snippet: Optional[str],
    raw_value: Optional[str] = None,
    extra_snippets: Optional[list[str]] = None,
) -> AttributeValue:
    snippets = [s for s in [snippet, *(extra_snippets or [])] if s]
    if value in (None, "", []) or not snippets:
        return not_mentioned()
    eids: list[str] = []
    for text in snippets:
        try:
            eids.append(
                store.add_from_structured(
                    product_id=product_id, attribute=attribute, value=value, status="stated",
                    raw_value=raw_value, evidence_text=text[:_MAX_SNIPPET],
                    source_url=page.record["product_url"], source_file=page.record["raw_html_path"],
                    raw_text=page.raw_text, html_text=page.html, extra_text=page.widget_text,
                )
            )
        except EvidenceIntegrityError:
            warnings.append(
                NormalizationWarning(
                    product_id=product_id, code="detail_evidence_not_verbatim", attribute=attribute,
                    message=f"Dropped an evidence snippet for {attribute}: not a verbatim substring of the page.",
                )
            )
    if not eids:
        return not_mentioned()
    return AttributeValue(value=value, status="stated", raw_value=raw_value, evidence_ids=eids)


def _first_int(text: str) -> Optional[int]:
    m = re.search(r"\d[\d,]*", text)
    return int(m.group(0).replace(",", "")) if m else None


def _line_around(text: str, m: re.Match, width: int = 80) -> str:
    """Snippet around a match, bounded by line breaks / quotes."""
    start = max(text.rfind("\n", 0, m.start()), text.rfind('"', 0, m.start()), m.start() - width) + 1
    ends = [i for i in (text.find("\n", m.end()), text.find('"', m.end())) if i != -1]
    end = min([m.end() + width, *ends])
    text_ = text[start:end]
    if start > 0 and text[start - 1].isalnum() and " " in text_:
        text_ = text_.split(" ", 1)[1]
    if end < len(text) and text[end].isalnum() and " " in text_:
        text_ = text_.rsplit(" ", 1)[0]
    return text_.strip()


def _trim_to_words(scope: str, m: re.Match) -> str:
    """The matched text without a partial word cut off at either end."""
    text, start, end = m.group(0), m.start(), m.end()
    if start > 0 and scope[start - 1].isalnum() and " " in text:
        text = text.split(" ", 1)[1]
    if end < len(scope) and scope[end].isalnum() and " " in text:
        text = text.rsplit(" ", 1)[0]
    return text.strip()


def _price_snippet(page: _Page) -> Optional[str]:
    d = page.details
    if page.brand == "Philips":
        sku = Path(page.record["raw_html_path"]).stem.split("_", 1)[1].upper().replace("-", "/")
        at = page.unescaped.find('"baseCommerceProduct":{"id":"' + sku + '"')
        if at < 0:
            return None
        m = re.compile(r'"price":\{[^{}]*\}(?:,"[A-Za-z]+":\{[^{}]*\}){0,3}').search(page.unescaped, at)
        return m.group(0) if m else None
    sale, orig = d.get("sale_price"), d.get("original_price")
    if sale is None:
        return None
    if orig is not None and orig != sale:
        m = re.search(rf"\${sale:,.2f}\n\${orig:,.2f}\nSave \$[\d,.]+", page.raw_text)
    else:
        m = re.search(rf"\${sale:,.2f}", page.raw_text)
        if not m:
            # Some pages omit the price from visible text; the product meta tag carries it.
            m = re.search(rf'<meta property="product:price:amount" content="{sale:.2f}"[^>]*>', page.html)
    return m.group(0) if m else None


def extract_detail_attributes(
    record: dict,
    product_id: str,
    store: EvidenceStore,
    warnings: list[NormalizationWarning],
) -> dict[str, AttributeValue]:
    out: dict[str, AttributeValue] = {name: not_mentioned() for name in DETAIL_ATTRIBUTES}
    page = _Page(record)
    d = page.details
    if not d:
        return out

    def add(attribute, value, snippet, raw_value=None, extra=None):
        out[attribute] = _add(page, store, warnings, product_id, attribute, value, snippet, raw_value, extra)

    # -- pricing -----------------------------------------------------------
    price_snip = _price_snippet(page)
    if price_snip and d.get("sale_price") is not None:
        pair = f"{d.get('original_price')} -> {d.get('sale_price')} {d.get('currency') or ''}".strip()
        add("original_price", d.get("original_price"), price_snip, pair)
        add("price_after_discount", d.get("sale_price"), price_snip, pair)
        add("discount_amount", d.get("discount_amount"), price_snip, pair)
        add("discount_pct", d.get("discount_pct"), price_snip, pair)
    promo = d.get("promotion_text")
    if promo:
        m = re.search(re.escape(promo[:60]) + r"[^\n]*", page.raw_text)
        add("promotion_text", promo, m.group(0) if m else None)

    # -- colors --------------------------------------------------------------
    colors = d.get("colors") or []
    if colors:
        snippets = []
        for color in colors:
            m = re.search(r'alt="[^"]*' + re.escape(color) + r'[^"]*"', page.html, re.I)
            if m:
                snippets.append(m.group(0))
        if not snippets and page.brand == "Philips":
            item = page.spec_item(r"colou?rs?")
            snippets = [item[2]] if item and item[2] else []
        if snippets:
            add("colors", colors, snippets[0], extra=snippets[1:])
            add("color_count", len(colors), snippets[0], extra=snippets[1:])

    # -- ratings (average and count only) --------------------------------------
    if d.get("rating_avg") is not None and d.get("rating_count") is not None:
        if page.brand == "Philips":
            m = re.search(r'"aggregateRating":\{[^{}]*\}', page.html)
            snippet = m.group(0) if m else None
        else:
            snippet = page.widget_text or None
        pair = f"{d['rating_avg']} / {d.get('rating_scale', 5)} from {d['rating_count']} ratings"
        add("rating_avg", d["rating_avg"], snippet, pair)
        add("rating_count", d["rating_count"], snippet, pair)

    # -- modes in detail -----------------------------------------------------------
    modes = d.get("modes") or []
    if modes:
        if page.brand == "Philips":
            item = page.chapters.get("Modes")
            snippet = item[1][:_MAX_SNIPPET] if item else None
        else:
            # The stated list: first mode name through the following text,
            # chosen at the occurrence where the second mode name follows.
            # (case-insensitive: names are title-cased, the page may not be)
            snippet = None
            second = modes[1]["name"].lower() if len(modes) > 1 else None
            for m in re.finditer(re.escape(modes[0]["name"]), page.raw_text, re.I):
                window = page.raw_text[m.start():m.start() + _MAX_SNIPPET]
                if second is None or second in window[:250].lower():
                    snippet = window
                    break
        add("mode_details", modes, snippet)
    if d.get("intensities"):
        item = page.chapters.get("Intensities") if page.brand == "Philips" else None
        if item is None and page.brand == "Philips":
            hit = page.spec_item(r"\d+ intensit(?:y|ies)")
            snippet = hit[2] if hit else None
        else:
            snippet = item[1][:_MAX_SNIPPET] if item else None
        add("intensity_levels", d["intensities"], snippet)

    # -- warranty, battery, charging, speed ------------------------------------------
    if page.brand == "Philips":
        hit = page.spec_item(r"Warranty")
        if hit and (y := re.search(r"(\d+)[\s-]*year", hit[1][0] if hit[1] else "", re.I)):
            add("warranty_years", int(y.group(1)), hit[2], hit[1][0])
        hit = page.spec_item(r"Battery type")
        if hit and hit[1]:
            add("battery_type", _normalize_battery_type(hit[1][0]), hit[2], hit[1][0])
        hit = page.spec_item(r"Speed")
        if hit and hit[1] and (n := _first_int(hit[1][0])) and n >= 1000:
            add("brush_movements_per_min", n, hit[2], hit[1][0])
    else:
        m = re.search(r"warranty starts on the day of purchase and runs for (\d+) years?", page.raw_text, re.I)
        if m:
            add("warranty_years", int(m.group(1)), m.group(0))
        m = re.search(r"[^\n.]*\blithium[- ]ion\b[^\n.]*", page.raw_text, re.I)
        if m:
            add("battery_type", "Lithium-ion", m.group(0).strip())
    if out["brush_movements_per_min"].status == "not_mentioned":
        m = page.search(r"([\d,]{4,})\s*(?:brush|bristle)?\s*(?:movements?|oscillations?|pulsations?|strokes?)(?:\s*(?:/|per|a)\s*min\w*)?")
        if m:
            scope = next(s for s in page.scopes if m.group(0) in s)
            add("brush_movements_per_min", _first_int(m.group(1)), _line_around(scope, m), m.group(0))
    m = page.search(r"[^\n.\"]{0,60}\bcharge in (?:just |only |about )?~?(\d+(?:\.\d+)?) hours?")
    if m:
        scope = next(s for s in page.scopes if m.group(0) in s)
        add("charge_time_hours", float(m.group(1)), m.group(0).strip(), m.group(0).strip())

    # -- feature flags (True only when the page states the feature) --------------------
    def flag(attribute: str, pattern: str):
        m = page.search(pattern)
        if m:
            scope = next(s for s in page.scopes if m.group(0) in s)
            add(attribute, True, _line_around(scope, m))

    if page.brand == "Philips":
        hit = page.spec_item(r"Display")
        if hit and hit[1]:
            add("display_type", hit[1][0], hit[2], hit[1][0])
    else:
        m = re.search(
            r"\b(?:(?:clear|bright)\s+)?(?:on[- ]handle|interactive|color|colour|LED)\s+"
            r"(?:(?:color|colour|black & white)\s+)?display\b",
            page.raw_text, re.I,
        )
        if m:
            add("display_type", m.group(0)[0].upper() + m.group(0)[1:].lower(), _line_around(page.raw_text, m))
    flag(
        "replacement_reminder",
        r"brush head tracker|replacement reminder|refill (?:change|replacement) (?:alert|reminder)|"
        r"(?:head|refill) replacement (?:reminder|alert|indicator)|BrushSync",
    )
    flag(
        "coverage_tracking",
        r"3D teeth tracking|teeth tracking|position detection|real-time coverage|brushing coverage|"
        r"pressure, motion,? and coverage|\bcoverage\b[^.\"]{0,30}(?:tracking|feedback|guidance)",
    )

    # -- included hardware -------------------------------------------------------------
    if page.brand == "Philips":
        hit = page.spec_item(r"Travel case", "Items included")
        if hit:
            add("travel_case_included", True, hit[2], hit[1][0] if hit[1] else None)
        hit = page.spec_item(r"Brush heads?", "Items included")
        if hit and hit[1] and (n := _first_int(hit[1][0])) is not None:
            add("brush_heads_included", n, hit[2], hit[1][0])
    else:
        # Only the page's own "What's in the box" list counts (never marketing
        # copy or internal tags): travel case present, brush heads summed.
        box = d.get("box_items") or []
        case_items = [i for i in box if re.search(r"\btravel (?:case|pouch)", i, re.I)]
        if case_items:
            add("travel_case_included", True, case_items[0], extra=case_items[1:])
        head_items = [i for i in box if re.search(r"\bbrush heads?\b", i, re.I)]
        leads = [re.match(r"\((\d+)\)", i.strip()) for i in head_items]
        if head_items and all(leads):
            add("brush_heads_included", sum(int(m.group(1)) for m in leads), head_items[0], extra=head_items[1:])
        elif head_items:
            # A head line without a "(N)" quantity: report no number rather than guess.
            warnings.append(NormalizationWarning(
                product_id=product_id, code="brush_head_quantity_not_stated", attribute="brush_heads_included",
                message=f"'What's in the box' lists brush heads without a quantity ({head_items}); no count reported.",
            ))
        if d.get("dimensions"):
            add("dimensions", d["dimensions"], d["dimensions"])
        if d.get("weight"):
            add("weight", d["weight"], d["weight"])

    m = re.search(r"(\d+)-day money-back guarantee", page.raw_text, re.I)
    if m:
        add("money_back_guarantee_days", int(m.group(1)), _line_around(page.raw_text, m, 40))

    # -- headline efficacy claims (verbatim text) ----------------------------------------
    figure = r"\d+\s*(?:x|%)"
    filler = r"[^.\n\"]"
    for attribute, topic in (("plaque_removal_claim", r"plaque"), ("gum_health_claim", r"healthier gums|gum health")):
        m = page.search(
            rf"{filler}{{0,80}}\b{figure}{filler}{{0,60}}\b(?:{topic})\b{filler}{{0,60}}|"
            rf"{filler}{{0,80}}\b(?:{topic})\b{filler}{{0,60}}{figure}{filler}{{0,60}}"
        )
        if m:
            scope = next(s for s in page.scopes if m.group(0) in s)
            text = _trim_to_words(scope, m)
            add(attribute, text, text)
    return out


def _normalize_battery_type(raw: str) -> str:
    low = raw.lower()
    if "lithium" in low:
        return "Lithium-ion"
    if "nimh" in low or "nickel" in low:
        return "NiMH"
    return raw.strip()


# ---------------------------------------------------------------------------
# Merge with the generic extractors' output
# ---------------------------------------------------------------------------

def _supersede(store: EvidenceStore, old: AttributeValue) -> None:
    for eid in old.evidence_ids:
        store.discard(eid)


def _timer_sentences(record: dict) -> list[str]:
    """Timer sentences from the shopper-visible copy, product description
    first, then FAQ answers; questions and navigation items are skipped."""
    from .utils import split_sentences

    ordered = sorted(
        record["sections"],
        key=lambda sec: 0 if sec["heading"] == "Product description" else 1,
    )
    picked: list[str] = []
    for section in ordered:
        if re.search(r"difference between|\bvs\.?\b", section["heading"], re.I):
            continue
        for line in section["text"].split("\n"):
            for sentence in split_sentences(line):
                sentence = sentence.strip()
                if (
                    len(sentence) >= 40
                    and not sentence.endswith("?")
                    and re.search(r"\btimer\b", sentence, re.I)
                    and re.search(r"\d", sentence)
                    and sentence not in picked
                    and sentence in record["raw_text"]
                ):
                    picked.append(sentence)
    return picked


def apply_detail_attributes(
    attrs: dict[str, AttributeValue],
    record: dict,
    product_id: str,
    store: EvidenceStore,
    warnings: list[NormalizationWarning],
) -> dict[str, AttributeValue]:
    """Add the detail attributes to ``attrs`` (the generic extractors'
    results keyed by NormalizedProduct field name) and let better-sourced
    values replace weaker ones: regular price as ``price``, structured
    modes, the box list as ``included_items``, a fuller Oral-B timer
    description. Also links disclaimers to the statements they qualify."""
    from .disclaimers import link_disclaimers

    detail = extract_detail_attributes(record, product_id, store, warnings)
    page = _Page(record)
    d = page.details
    out = dict(attrs)
    out.update(detail)
    if not d:
        return out

    def add(attribute, value, snippet, raw_value=None, extra=None):
        return _add(page, store, warnings, product_id, attribute, value, snippet, raw_value, extra)

    # -- regular price is the primary price -------------------------------------
    reg = detail["original_price"]
    if reg.status == "stated":
        snippet = _price_snippet(page)
        check_text = "\n".join((page.raw_text, page.html, page.unescaped, page.widget_text))
        if snippet and snippet in check_text:
            _supersede(store, attrs["price"])
            out["price"] = add("price", reg.value, snippet, reg.raw_value)

    # -- modes: structured names (and Oral-B's stated count) -------------------------
    modes = d.get("modes") or []
    if modes and attrs["modes"].status == "not_mentioned" and detail["mode_details"].status == "stated":
        snippet = store.get(detail["mode_details"].evidence_ids[0]).evidence_text
        out["modes"] = add("modes", [m["name"] for m in modes], snippet)
    if page.brand == "Oral-B" and d.get("mode_count") and attrs["mode_count"].status == "not_mentioned":
        n = d["mode_count"]
        m = re.search(rf"\b{n}\s+(?:[A-Za-z/\-]+\s+){{0,3}}?modes?\b[^\n]*", page.raw_text, re.I)
        if m:
            out["mode_count"] = add("mode_count", n, m.group(0).strip(), m.group(0).strip())

    # -- Oral-B: fuller timer description from the visible copy --------------------------
    if page.brand == "Oral-B" and attrs["timer"].status == "stated":
        sentences = _timer_sentences(record)[:2]
        if sentences:
            _supersede(store, attrs["timer_description"])
            out["timer_description"] = add(
                "timer_description", " ".join(sentences), sentences[0], extra=sentences[1:]
            )

    # -- Oral-B: included items = the "What's in the box" list -----------------------------
    if page.brand == "Oral-B" and d.get("box_items"):
        idx = next((i for i, s_ in enumerate(record["sections"]) if s_["heading"] == "What's in the box"), None)
        if idx is not None:
            from .utils import combined_section_text

            section = record["sections"][idx]
            combined = combined_section_text(section)
            items = [i for i in d["box_items"] if i in combined]
            if items:
                _supersede(store, attrs["included_items"])
                eids = [
                    store.add_from_section(
                        product_id=product_id, attribute="included_items", value=item, status="stated",
                        raw_value=item, evidence_text=item,
                        source_url=record["product_url"], source_file=record["raw_html_path"],
                        section_heading=section["heading"], section_index=idx,
                        section_combined_text=combined,
                    )
                    for item in items
                ]
                out["included_items"] = AttributeValue(value=items, status="stated", raw_value=None, evidence_ids=eids)

    from .detail_extras import apply_extras

    return apply_extras(out, record, product_id, store, warnings)
