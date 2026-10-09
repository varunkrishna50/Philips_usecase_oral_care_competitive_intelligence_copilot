"""Second layer of detail attributes (review round 2): the specifics behind
the boolean columns, battery/charging/timer/pressure/app prose, own-product
facts from comparison FAQs, Philips specification rows, availability and the
claim <-> disclaimer linking.

Runs after ``detail_attributes.extract_detail_attributes`` and may *replace*
the generic extractors' weaker values (their evidence is discarded so it never
reaches evidence.json). Same contract as everywhere in Step 2: every stated
value has evidence that is a verbatim substring of the page; absence is never
inferred.
"""
from __future__ import annotations

import html
import re
from typing import Any, Optional

from .detail_attributes import _add, _Page, _price_snippet, _supersede
from .disclaimers import link_disclaimers, marker_runs
from .evidence import EvidenceStore
from .filters import usable_sections
from .models import AttributeValue, NormalizationWarning, not_mentioned
from .topics import TOPICS, TopicSentence, collect_topic, own_side_facts
from .utils import combined_section_text, split_sentences

EXTRA_ATTRIBUTES = [
    "availability",
    "listed_price_hidden",
    "pressure_sensing_details",
    "app_connectivity_details",
    "bluetooth_details",
    "battery_details",
    "charging_details",
    "coverage_tracking_details",
    "brushing_zones",
    "zone_interval_seconds",
    "brush_head_replacement_interval",
    "cleaning_technology",
    "brush_head_compatibility",
    "feel_and_noise",
    "price_positioning",
    "noise_db",
    "intensity_details",
    "intensity_count",
    "battery_indicator",
]

_PRICE_GROUP = ("price", "original_price", "price_after_discount", "discount_amount", "discount_pct")


class _Ctx:
    def __init__(self, out, record, product_id, store, warnings):
        self.out: dict[str, AttributeValue] = out
        self.record = record
        self.product_id = product_id
        self.store = store
        self.warnings = warnings
        self.page = _Page(record)
        self.d = self.page.details
        self.sections = usable_sections(record["sections"], product_id, [])

    # -- evidence helpers -------------------------------------------------------
    def structured(self, attribute, value, snippet, raw_value=None, extra=None) -> AttributeValue:
        return _add(self.page, self.store, self.warnings, self.product_id, attribute, value, snippet, raw_value, extra)

    def from_section(self, attribute: str, value: Any, text: str, section_index: int, raw_value=None) -> Optional[str]:
        section = self.record["sections"][section_index]
        combined = combined_section_text(section)
        if text not in combined:
            return None
        return self.store.add_from_section(
            product_id=self.product_id, attribute=attribute, value=value, status="stated", raw_value=raw_value,
            evidence_text=text, source_url=self.record["product_url"], source_file=self.record["raw_html_path"],
            section_heading=section["heading"], section_index=section_index, section_combined_text=combined,
        )

    def replace(self, attribute: str, new: AttributeValue) -> None:
        if new.status == "not_mentioned":
            return
        old = self.out.get(attribute)
        if old is not None and old.evidence_ids and old is not new:
            _supersede(self.store, old)
        self.out[attribute] = new

    def sentences_attr(self, attribute: str, sentences: list[TopicSentence], as_text: bool = False) -> AttributeValue:
        eids, texts = [], []
        for s in sentences:
            eid = self.from_section(attribute, s.text, s.text, s.section_index)
            if eid:
                eids.append(eid)
                texts.append(s.text)
        if not eids:
            return not_mentioned()
        value: Any = " ".join(texts) if as_text else texts
        return AttributeValue(value=value, status="stated", raw_value=None, evidence_ids=eids)


# --- Philips: spec rows and feature panels as topic sources -----------------------------

_PH_SPEC_CHAPTERS_FOR_TOPICS = ("Ease of use", "Smart sensor technology", "Sonicare app", "Technical specifications", "Power")


def _philips_features(page: _Page) -> list[dict]:
    region = getattr(page, "features_region", None)
    if not region:
        return []
    start = region.find("[")
    try:
        import json

        return json.JSONDecoder().raw_decode(region, start)[0]
    except ValueError:
        return []


def _philips_topic(ctx: _Ctx, topic: str, limit: int = 6) -> list[tuple[str, str]]:
    """[(display_text, verbatim_snippet)] for a topic from feature panels + spec rows."""
    spec = TOPICS[topic]
    sentence_re = re.compile(spec["sentence"], re.I)
    heading_re = re.compile(spec["heading"], re.I)
    page = ctx.page
    out: list[tuple[str, str]] = []
    for item in _philips_features(page):
        name = str(item.get("featureName") or "")
        whole = bool(heading_re.search(name))
        long_desc = str(item.get("featureLongDescription") or "").strip()
        texts = ([long_desc] if (len(long_desc) >= 60 or re.search(r"[.!]", long_desc)) else []) + split_sentences(
            str(item.get("featureGlossary") or "")
        )
        for sentence in texts:
            sentence = sentence.strip()
            if len(sentence) < 20 or sentence not in page.unescaped:
                continue
            if sentence_re.search(sentence) or (whole and re.search(r"\d", sentence)):
                out.append((sentence, sentence))
    for cname, (obj, region) in page.chapters.items():
        if cname not in _PH_SPEC_CHAPTERS_FOR_TOPICS:
            continue
        for item in obj.get("csItem", []):
            iname = str(item.get("csItemName", ""))
            values = [str(v.get("csValueName", "")).strip() for v in item.get("csValue", []) if v.get("csValueName")]
            if not values:
                continue
            row = f"{iname}: {'; '.join(values)}"
            if sentence_re.search(row) or heading_re.search(iname):
                hit = page.spec_item(re.escape(iname), cname)
                if hit and hit[2]:
                    out.append((row, hit[2]))
    seen, uniq = set(), []
    for text, snippet in out:
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if key not in seen:
            seen.add(key)
            uniq.append((text, snippet))
    return uniq[:limit]


# --- main entry ----------------------------------------------------------------------------

def apply_extras(
    out: dict[str, AttributeValue],
    record: dict,
    product_id: str,
    store: EvidenceStore,
    warnings: list[NormalizationWarning],
) -> dict[str, AttributeValue]:
    for name in EXTRA_ATTRIBUTES:
        out.setdefault(name, not_mentioned())
    ctx = _Ctx(out, record, product_id, store, warnings)
    if not ctx.d:
        return out
    _availability_and_price(ctx)
    _description(ctx)
    _modes(ctx)
    _topic_details(ctx)
    _own_side(ctx)
    _brush_head_fit(ctx)
    _brush_head_replacement(ctx)
    _claims_sentences(ctx)
    if ctx.page.brand == "Philips":
        _philips_specs(ctx)
    _disclaimers(ctx)
    return ctx.out


# --- availability + price visibility ------------------------------------------------------------

def _availability_and_price(ctx: _Ctx) -> None:
    d, page = ctx.d, ctx.page
    state = d.get("availability")
    if not state:
        return
    label = {"in_stock": "In stock", "out_of_stock": "Out of stock"}.get(state, state)
    if page.brand == "Philips":
        m = re.search(r'"availability":\{"status":"[A-Z_]+"', page.unescaped)
    else:
        m = re.search(r'"availability"\s*:\s*"https://schema\.org/[A-Za-z]+"', page.html)
    if m:
        ctx.out["availability"] = ctx.structured("availability", label, m.group(0), m.group(0))
    if state == "out_of_stock" and d.get("price_displayed") is False:
        hidden = ctx.out["original_price"].value if ctx.out["original_price"].status == "stated" else None
        for name in _PRICE_GROUP:
            if ctx.out[name].status == "stated":
                _supersede(ctx.store, ctx.out[name])
                ctx.out[name] = not_mentioned()
        snippet = _price_snippet(page)
        if hidden is not None and snippet:
            # Keep the number, but only as an explicitly-labelled hidden value.
            ctx.out["listed_price_hidden"] = ctx.structured(
                "listed_price_hidden", hidden, snippet,
                "listed in page data only; the page shows no price while the product is out of stock",
            )
        ctx.warnings.append(NormalizationWarning(
            product_id=ctx.product_id, code="price_hidden_out_of_stock", attribute="price",
            message="The page shows no price because the product is out of stock; the listed number is kept only in listed_price_hidden.",
        ))


# --- description: the visible lead paragraph ---------------------------------------------------------

def _description(ctx: _Ctx) -> None:
    text = ctx.d.get("short_description")
    if not text:
        return
    for idx, section in ctx.sections:
        for line in section["text"].split("\n"):
            if line.strip() == text or line.strip().startswith(text[:80]):
                line = line.strip()
                eid = ctx.from_section("description", line, line, idx, raw_value=line)
                if eid:
                    ctx.replace("description", AttributeValue(value=line, status="stated", raw_value=line, evidence_ids=[eid]))
                    return


# --- modes: structured names always win -----------------------------------------------------------------

def _modes(ctx: _Ctx) -> None:
    modes = ctx.d.get("modes") or []
    detail = ctx.out.get("mode_details")
    if not modes or detail is None or detail.status != "stated":
        return
    snippet = ctx.store.get(detail.evidence_ids[0]).evidence_text
    new = ctx.structured("modes", [m["name"] for m in modes], snippet)
    ctx.replace("modes", new)


# --- topic prose ----------------------------------------------------------------------------------------

def _topic_details(ctx: _Ctx) -> None:
    out = ctx.out
    brand = ctx.page.brand

    def run(topic: str, attribute: str, gate: Optional[str] = None, as_text: bool = False) -> None:
        if gate and out[gate].status != "stated":
            return
        if brand == "Philips":
            items = _philips_topic(ctx, topic, 6 if topic not in ("timer", "pressure") else 8)
            if items:
                av = ctx.structured(attribute, [t for t, _ in items] if not as_text else " ".join(t for t, _ in items),
                                    items[0][1], None, [s for _, s in items[1:]])
                ctx.replace(attribute, av)
        else:
            sentences = collect_topic(ctx.sections, topic)
            if sentences:
                ctx.replace(attribute, ctx.sentences_attr(attribute, sentences, as_text))

    run("pressure", "pressure_sensing_details", gate="pressure_sensing")
    run("app", "app_connectivity_details", gate="app_connectivity")
    run("bluetooth", "bluetooth_details", gate="bluetooth_connectivity")
    run("battery", "battery_details")
    run("charging", "charging_details")
    run("coverage", "coverage_tracking_details", gate="coverage_tracking")
    # timer: a full description replaces the single generic sentence
    if out["timer"].status == "stated":
        run("timer", "timer_description", as_text=True)

    # charge time (hours): "takes about 3 hours", "3 hour fast charger", "full charge in ~3 hours"
    if out["charge_time_hours"].status != "stated":
        pool = collect_topic(ctx.sections, "battery") + collect_topic(ctx.sections, "charging")
        rx = re.compile(
            r"(?:takes?|in|within)\s+(?:just\s+|only\s+|about\s+|around\s+|approximately\s+|~)?(\d+(?:\.\d+)?)\s*hours?|"
            r"(\d+(?:\.\d+)?)[- ]hours?\s+(?:fast\s+)?charg",
            re.I,
        )
        for s in pool:
            m = rx.search(s.text)
            if m:
                value = float(m.group(1) or m.group(2))
                eid = ctx.from_section("charge_time_hours", value, s.text, s.section_index, raw_value=m.group(0))
                if eid:
                    out["charge_time_hours"] = AttributeValue(value=value, status="stated", raw_value=m.group(0), evidence_ids=[eid])
                    break

    # pacing: zone interval / zone count
    timer_sentences = collect_topic(ctx.sections, "timer") if brand != "Philips" else []
    for s in timer_sentences:
        m = re.search(r"every (\d+)[- ]seconds?", s.text, re.I)
        if m and out["zone_interval_seconds"].status != "stated":
            eid = ctx.from_section("zone_interval_seconds", int(m.group(1)), s.text, s.section_index, raw_value=m.group(0))
            if eid:
                out["zone_interval_seconds"] = AttributeValue(value=int(m.group(1)), status="stated", raw_value=m.group(0), evidence_ids=[eid])
        zones = re.search(r"\b(quadrant|sextant)s?\b", s.text, re.I)
        if zones and out["brushing_zones"].status != "stated":
            n = 4 if zones.group(1).lower() == "quadrant" else 6
            eid = ctx.from_section("brushing_zones", n, s.text, s.section_index, raw_value=zones.group(0))
            if eid:
                out["brushing_zones"] = AttributeValue(value=n, status="stated", raw_value=zones.group(0), evidence_ids=[eid])


# --- facts about this product from comparison FAQs ------------------------------------------------------------

_OWN_FIELD_TO_ATTR = {
    "cleaning_technology": "cleaning_technology",
    "brush_head_compatibility": "brush_head_compatibility",
    "feel_and_noise": "feel_and_noise",
    "price_positioning": "price_positioning",
    "display_type": "display_type",
}


def _own_side(ctx: _Ctx) -> None:
    if ctx.page.brand != "Oral-B":
        return
    facts = own_side_facts(list(enumerate(ctx.record["sections"])), ctx.record["product_name"], ctx.product_id)
    for fact in facts:
        field, text, idx = fact["field"], fact["text"], fact["section_index"]
        basis = f" (comparison FAQ vs {fact['compared_with']})" if fact["compared_with"] else ""
        if field in _OWN_FIELD_TO_ATTR:
            attr = _OWN_FIELD_TO_ATTR[field]
            if ctx.out[attr].status == "stated":
                continue
            eid = ctx.from_section(attr, text, text, idx, raw_value=f"from comparison FAQ{basis}")
            if eid:
                ctx.out[attr] = AttributeValue(value=text, status="stated", raw_value=f"from comparison FAQ{basis}", evidence_ids=[eid])
                if attr == "feel_and_noise":
                    m = re.search(r"(\d+)\s*dB", text, re.I)
                    if m:
                        e2 = ctx.from_section("noise_db", int(m.group(1)), text, idx, raw_value=m.group(0))
                        if e2:
                            ctx.out["noise_db"] = AttributeValue(value=int(m.group(1)), status="stated", raw_value=m.group(0), evidence_ids=[e2])
        elif field == "pacing":
            m = re.search(r"(\d+)[- ]second\s+(quadrant|sextant)\s+intervals?\s*\((\d+)\s*zones?\)", text, re.I)
            if m:
                for attr, value, raw in (("zone_interval_seconds", int(m.group(1)), m.group(0)), ("brushing_zones", int(m.group(3)), m.group(0))):
                    eid = ctx.from_section(attr, value, text, idx, raw_value=raw)
                    if eid:
                        ctx.out[attr] = AttributeValue(value=value, status="stated", raw_value=raw, evidence_ids=[eid])
        elif field == "box_note":
            ctx.warnings.append(NormalizationWarning(
                product_id=ctx.product_id, code="box_quantity_statement_conflicts_with_box_list", attribute="brush_heads_included",
                message=f"A comparison FAQ says {text!r}; compare with the 'What's in the box' list before relying on brush head counts.",
            ))


def _brush_head_fit(ctx: _Ctx) -> None:
    """"Fits all Oral-B iO replacement brush heads: ..." when no comparison FAQ gave it."""
    if ctx.out["brush_head_compatibility"].status == "stated":
        return
    rx = re.compile(r"\b(?:fits all|compatible (?:only )?with)\b[^.]*brush heads?[^.]*\.?", re.I)
    for idx, section in ctx.sections:
        for line in section["text"].split("\n"):
            m = rx.search(line)
            if m and "?" not in line:
                text = m.group(0).strip()
                eid = ctx.from_section("brush_head_compatibility", text, text, idx)
                if eid:
                    ctx.out["brush_head_compatibility"] = AttributeValue(value=text, status="stated", raw_value=None, evidence_ids=[eid])
                    return


# --- brush head replacement advice -------------------------------------------------------------------------------

def _brush_head_replacement(ctx: _Ctx) -> None:
    rx = re.compile(r"(?:replac\w+|change)[^.]{0,60}brush head[^.]{0,40}every\s+(?:\d+|three|3)\s+months?[^.]*\.?|"
                    r"every\s+\d+\s+months?\s+or\s+sooner[^.]*\.?", re.I)
    for idx, section in ctx.sections:
        for sentence in split_sentences(section["text"].replace("\n", " \n")):
            sentence = sentence.strip()
            if rx.search(sentence) and "?" not in sentence:
                eid = ctx.from_section("brush_head_replacement_interval", sentence, sentence, idx)
                if eid:
                    ctx.out["brush_head_replacement_interval"] = AttributeValue(
                        value=sentence, status="stated", raw_value=None, evidence_ids=[eid])
                    return


# --- plaque / gum claims: the whole enclosing sentence (Oral-B) ---------------------------------------------------------

def _claims_sentences(ctx: _Ctx) -> None:
    if ctx.page.brand != "Oral-B":
        return
    figure = re.compile(r"\d+\s*(?:x|%)", re.I)
    for attribute, topic in (("plaque_removal_claim", re.compile(r"\bplaque\b", re.I)),
                             ("gum_health_claim", re.compile(r"healthier gums|gum health", re.I))):
        best = None
        for idx, section in ctx.sections:
            if re.search(r"warranty|featured", section["heading"], re.I):
                continue
            for sentence in _claim_sentences(section["text"]):
                if figure.search(sentence) and topic.search(sentence) and "?" not in sentence:
                    score = (bool(re.search(r"\bvs\.?|versus|compared", sentence, re.I)), -len(sentence))
                    if best is None or score > best[0]:
                        best = (score, sentence, idx)
        if best:
            sentence, idx = best[1], best[2]
            eid = ctx.from_section(attribute, sentence, sentence, idx)
            if eid:
                ctx.replace(attribute, AttributeValue(value=sentence, status="stated", raw_value=None, evidence_ids=[eid]))


def _claim_sentences(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        for sentence in split_sentences(line):
            sentence = sentence.strip()
            # drop a glued tab/label prefix such as "More Information Switch to ..."
            sentence = re.sub(r"^(?:More Information|What's in the box|Make the most of every session)\s+", "", sentence)
            if 25 <= len(sentence) <= 400:
                out.append(sentence)
    return out


# --- Philips specification rows ---------------------------------------------------------------------------------------------

def _philips_specs(ctx: _Ctx) -> None:
    page, out = ctx.page, ctx.out
    chapters = page.chapters

    def rows(chapter: str) -> list[tuple[str, list[str], str]]:
        found = []
        if chapter not in chapters:
            return found
        for item in chapters[chapter][0].get("csItem", []):
            iname = str(item.get("csItemName", ""))
            hit = page.spec_item(re.escape(iname), chapter)
            if hit and hit[1]:
                found.append((iname, hit[1], hit[2]))
        return found

    # battery life (days) from the "Operating time" spec row
    if out["battery_life_days"].status != "stated":
        for iname, values, snippet in rows("Technical specifications"):
            m = re.fullmatch(r"(?:operating time.*|battery life.*)", iname, re.I)
            n = re.search(r"(\d+)\s*(days?|weeks?)", values[0], re.I)
            if m and n:
                days = int(n.group(1)) * (7 if n.group(2).lower().startswith("week") else 1)
                ctx.replace("battery_life_days", ctx.structured("battery_life_days", days, snippet, values[0]))
                ctx.replace("battery_life_raw", ctx.structured("battery_life_raw", f"{iname}: {values[0]}", snippet, values[0]))
                break

    # included items
    box = rows("Items included")
    if box:
        texts = [f"{iname}: {'; '.join(values)}" for iname, values, _ in box]
        eids = []
        for text, (_, _, snippet) in zip(texts, box):
            av = ctx.structured("included_items", texts, snippet)
            eids.extend(av.evidence_ids)
        if eids:
            ctx.replace("included_items", AttributeValue(value=texts, status="stated", raw_value=None, evidence_ids=eids))
        # charger => charging method
        if any(re.fullmatch(r"chargers?|charging.*", iname, re.I) for iname, _, _ in box) and out["charging_method"].value is None:
            iname, values, snippet = next(r for r in box if re.fullmatch(r"chargers?|charging.*", r[0], re.I))
            ctx.replace("charging_method", ctx.structured("charging_method", "charger", snippet, f"{iname}: {values[0]}"))
            ctx.replace("charging_method_raw", ctx.structured("charging_method_raw", f"{iname}: {values[0]}", snippet, values[0]))

    # battery indicator
    for iname, values, snippet in rows("Ease of use"):
        if re.fullmatch(r"battery indicator", iname, re.I):
            ctx.replace("battery_indicator", ctx.structured("battery_indicator", values[0], snippet, f"{iname}: {values[0]}"))

    # specifications = the actual spec rows
    skip = {"Modes", "Intensities", "Items included", "Sonicare app", "Cleaning performance", "Smart sensor technology"}
    texts, eids = [], []
    for cname in chapters:
        if cname in skip:
            continue
        for iname, values, snippet in rows(cname):
            text = html.unescape(f"{cname}: {iname} - {'; '.join(values)}")
            av = ctx.structured("specifications", text, snippet)
            if av.status == "stated":
                texts.append(text)
                eids.extend(av.evidence_ids)
    if texts:
        ctx.replace("specifications", AttributeValue(value=texts, status="stated", raw_value=None, evidence_ids=eids))

    # intensities: the feature text is authoritative when it counts more than the spec table
    spec_rows = rows("Intensities")
    spec_names = [iname for iname, _, _ in spec_rows]
    count, names, snippet = None, [], None
    for item in _philips_features(page):
        gl = str(item.get("featureGlossary") or "")
        m = re.search(r"(\d+)\s+intensity\s+settings?[^.]*\.\s*(?:Choose|Select)\s+between\s+([^.]+)\.", gl, re.I)
        if m and m.group(0) in page.unescaped:
            count = int(m.group(1))
            names = [n.strip() for n in re.split(r",\s*|\s+and\s+", m.group(2)) if n.strip()]
            snippet = m.group(0)
            break
    if count is None:
        for hl in ctx.d.get("highlights") or []:
            m = re.fullmatch(r"(\d+) intensit(?:y|ies)", hl.strip(), re.I)
            if m:
                count = int(m.group(1))
    if count is not None:
        if snippet:
            ctx.replace("intensity_levels", ctx.structured("intensity_levels", names, snippet))
        ctx.replace("intensity_count", ctx.structured("intensity_count", count, snippet or f"{count} intensities", snippet or None) if snippet else not_mentioned())
        if spec_names and len(spec_names) != count:
            ctx.warnings.append(NormalizationWarning(
                product_id=ctx.product_id, code="intensity_count_mismatch", attribute="intensity_levels",
                message=f"The feature text says {count} intensities ({', '.join(names)}) but the specification table lists {len(spec_names)} ({', '.join(spec_names)}); the feature text is used.",
            ))
    if spec_rows:
        details = [{"name": iname, "description": "; ".join(values)} for iname, values, _ in spec_rows]
        eids = []
        for _, _, snip in spec_rows:
            av = ctx.structured("intensity_details", details, snip)
            eids.extend(av.evidence_ids)
        if eids:
            ctx.replace("intensity_details", AttributeValue(value=details, status="stated", raw_value=None, evidence_ids=eids))
        if count is None:
            ctx.replace(
                "intensity_levels",
                ctx.structured("intensity_levels", spec_names, spec_rows[0][2], None, [r[2] for r in spec_rows[1:]]),
            )


# --- disclaimers: only claim-related footnotes, merged, with qualifiers --------------------------------------------------------------

_QUALIFIERS = [
    ("comparator", re.compile(r"(?:\bvs\.?|\bversus|compared (?:to|with))\s+(?:a |an |the )?[A-Za-z\- ]{3,40}?(?=[,.;:*)]|\s+(?:in|with|and|for|within)\b|$)", re.I)),
    ("timeframe", re.compile(r"\bin (?:just |only )?(?:less than )?\d+\s*(?:weeks?|days?|months?|minutes?)\b|\bwithin \d+\s*\w+|\bfrom day one\b", re.I)),
    ("evidence", re.compile(r"\bclinically proven\b|\bclinical(?:ly)? (?:study|tested)\b", re.I)),
]


def _qualifiers(claim: str) -> list[dict]:
    found, seen = [], set()
    for kind, rx in _QUALIFIERS:
        for m in rx.finditer(claim):
            text = m.group(0).strip(" ,.;*")
            if text and text.lower() not in seen and text in claim:
                seen.add(text.lower())
                found.append({"type": kind, "text": text})
    return found


def _norm(text: str) -> set[str]:
    stop = {"a", "an", "the", "regular", "standard"}
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop}


def _merge_disclaimers(raw: list[dict]) -> list[dict]:
    """Drop section-scoped footnotes (the warranty tab's '*'), then merge
    footnotes with the same marker whose wording overlaps."""
    kept: list[dict] = []
    for d in raw:
        if d.get("source") == "section":
            continue
        merged = False
        for i, k in enumerate(kept):
            if (k.get("marker") or "") != (d.get("marker") or ""):
                continue
            a, b = _norm(k["text"]), _norm(d["text"])
            if a and b and len(a & b) / min(len(a), len(b)) >= 0.5:
                if len(d["text"]) > len(k["text"]):
                    kept[i] = d
                merged = True
                break
        if not merged:
            kept.append(d)
    return kept


def _where(attribute: str, label: Optional[str]) -> str:
    if label:
        if label.startswith("Feature: "):
            return f"Feature panel '{label[len('Feature: '):]}'"
        if ": " in label:
            chapter, item = label.split(": ", 1)
            return f"Technical specifications > {chapter} > {item}"
        return label
    return {"claims": "Claims", "plaque_removal_claim": "Plaque removal claim", "gum_health_claim": "Gum health claim"}.get(attribute, attribute)


def _disclaimers(ctx: _Ctx) -> None:
    d, out = ctx.d, ctx.out
    raw = d.get("disclaimers") or []
    for name in ("disclaimers", "claims_with_disclaimers"):
        if out[name].status == "stated":
            _supersede(ctx.store, out[name])
            out[name] = not_mentioned()
    statements: list[dict] = []
    for attribute in ("claims", "plaque_removal_claim", "gum_health_claim"):
        value = out[attribute].value
        for text in (value if isinstance(value, list) else [value]):
            if isinstance(text, str) and text:
                statements.append({"attribute": attribute, "text": text, "where": _where(attribute, None)})
    for ms in d.get("marked_statements") or []:
        if ms["label"] == "Warranty information":
            continue  # warranty footnotes are not claim disclaimers
        statements.append({"attribute": ms["label"], "text": ms["text"], "where": _where("", ms["label"]), "section": ms["label"]})

    disclaimers = _merge_disclaimers(raw)
    linked = link_disclaimers(disclaimers, statements)

    entries, eids = [], []
    for item in linked["disclaimers"]:
        av = ctx.structured("disclaimers", item["text"], item["text"])
        if av.status != "stated":
            continue
        eids.extend(av.evidence_ids)
        where_by_text = {s["text"]: s["where"] for s in statements}
        entries.append({
            "marker": item.get("marker", ""), "text": item["text"], "source": item.get("source", "product"),
            "applies_to": [{"where": where_by_text.get(a["text"], a["attribute"]), "text": a["text"]} for a in item["applies_to"]],
        })
    if entries:
        out["disclaimers"] = AttributeValue(value=entries, status="stated", raw_value=None, evidence_ids=eids)

    claims = out["claims"]
    if claims.status == "stated" and isinstance(claims.value, list):
        rows, snippets = [], []
        for claim in claims.value:
            texts = linked["per_statement"].get(claim, [])
            quals = _qualifiers(claim)
            if not texts and not quals:
                continue
            rows.append({"claim": claim, "markers": marker_runs(claim), "disclaimers": texts, "qualifiers": quals})
            snippets.append(claim)
            snippets.extend(t for t in texts if t not in snippets)
        if rows:
            out["claims_with_disclaimers"] = ctx.structured("claims_with_disclaimers", rows, snippets[0], None, snippets[1:])

    for text in linked["orphan_disclaimers"]:
        ctx.warnings.append(NormalizationWarning(
            product_id=ctx.product_id, code="unlinked_disclaimer", attribute="disclaimers",
            message=f"No statement on the page carries the marker of disclaimer {text!r}.",
        ))
    for marker in linked["ambiguous_markers"]:
        ctx.warnings.append(NormalizationWarning(
            product_id=ctx.product_id, code="ambiguous_disclaimer_marker", attribute="disclaimers",
            message=f"Marker {marker!r} is used by more than one disclaimer; statements with it are linked to all of them.",
        ))
    for text, marker in linked["markers_without_disclaimer"]:
        ctx.warnings.append(NormalizationWarning(
            product_id=ctx.product_id, code="marker_without_disclaimer", attribute="disclaimers",
            message=f"Statement {text[:80]!r} carries marker {marker!r} but the page lists no matching disclaimer.",
        ))
