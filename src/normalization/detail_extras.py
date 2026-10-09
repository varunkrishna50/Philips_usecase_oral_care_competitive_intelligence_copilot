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

from .detail_attributes import _add, _Page, _price_snippets, _supersede
from .disclaimers import link_disclaimers, marker_runs
from .evidence import EvidenceStore
from .filters import usable_sections
from .models import AttributeValue, NormalizationWarning, not_mentioned
from .topics import TOPICS, TopicSentence, collect_topic, fold, own_side_facts, product_tokens
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
    "brush_head_design",
    "battery_life_note",
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
            if sentence_re.search(fold(sentence)) or (whole and re.search(r"\d", sentence)):
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
    _mode_count(ctx)
    if ctx.page.brand == "Philips":
        # flags first: the *_details prose below is gated on the flag being stated, and a
        # Philips flag is often only found in a feature panel / spec row (hx7410, hx7421, hx9990 app)
        _philips_flags(ctx)
    _topic_details(ctx)
    _technology(ctx)
    _intensity_count(ctx)
    _replacement_reminder(ctx)
    _battery_note(ctx)
    if ctx.page.brand == "Philips":
        _philips_replacement(ctx)
    _own_side(ctx)
    _display_rule(ctx)
    _brush_head_fit(ctx)
    _brush_head_replacement(ctx)
    _claims_sentences(ctx)
    if ctx.page.brand == "Philips":
        _philips_specs(ctx)
        _philips_rendered_fallbacks(ctx)
        _philips_claims(ctx)
    _disclaimers(ctx)
    return ctx.out


# --- display_type: a light ring / indicator is not a display ----------------------------------------------

def _display_rule(ctx: _Ctx) -> None:
    av = ctx.out["display_type"]
    if av.status != "stated" or not isinstance(av.value, str):
        return
    body = re.sub(r"^\s*display\s*/\s*indicator\s*:\s*", "", av.value, flags=re.I)
    if not re.search(r"\b(?:display|screen)\b", body, re.I):
        _supersede(ctx.store, av)
        ctx.out["display_type"] = not_mentioned()


# --- Philips claims: qualitative claims, spec-row claims, headline plaque/gum claims -------------------------------

_QUALITATIVE_CLAIM_RX = re.compile(
    r"recommended sonic toothbrush brand by dental professionals worldwide|"
    r"cleans your teeth better|"
    r"specially designed to protect kids.{1,2} teeth and gums|"
    r"buddy up, brush together, build great habits",
    re.I,
)
_FIGURE = re.compile(r"\d+(?:\.\d+)?\s*(?:x|%)", re.I)


def _philips_claims(ctx: _Ctx) -> None:
    """Round out the Philips claim list from visible text:
    * qualitative claims on pages with no figure claims (One, One For Kids);
    * "Cleaning performance" spec rows whose figure no other claim states
      (the 4100's 'Up to 5x more plaque removal*');
    then pick the shortest headline plaque / gum claim from the list."""
    out = ctx.out
    cur = out["claims"]
    values = list(cur.value) if cur.status == "stated" and isinstance(cur.value, list) else []
    eids = list(cur.evidence_ids) if cur.status == "stated" else []
    added = False

    # qualitative claims (sentence or heading), once each
    for idx, section in ctx.sections:
        candidates = [section["heading"]] + [s for line in section["text"].split("\n") for s in split_sentences(line)]
        for text in candidates:
            text = text.strip()
            m = _QUALITATIVE_CLAIM_RX.search(text)
            if not m or len(text) > 200 or "?" in text:
                continue
            if any(fold(text) == fold(v) for v in values):
                continue
            eid = ctx.from_section("claims", text, text, idx)
            if eid:
                values.append(text)
                eids.append(eid)
                added = True

    # spec-row claims with a figure not already stated elsewhere
    def topic_of(t: str) -> str:
        low = t.lower()
        return "plaque" if "plaque" in low else "gums" if "gum" in low else "stain" if re.search(r"stain|whit", low) else "other"

    def figure_keys(text: str) -> set[tuple[str, str]]:
        figs = list(_FIGURE.finditer(text))
        keys = set()
        for i, m in enumerate(figs):
            clause = text[m.start(): figs[i + 1].start() if i + 1 < len(figs) else len(text)]
            prefix = text[figs[i - 1].end() if i else 0: m.start()]
            keys.add((topic_of(clause) if topic_of(clause) != "other" else topic_of(prefix + clause),
                      m.group(0).lower().replace(" ", "")))
        return keys

    seen = set().union(*[figure_keys(v) for v in values]) if values else set()
    for iname, row_values, _snippet in _spec_rows(ctx, "Cleaning performance"):
        for value in row_values:
            m = _FIGURE.search(value)
            if not m:
                continue
            key = (topic_of(f"{iname} {value}"), m.group(0).lower().replace(" ", ""))
            if key in seen:
                continue
            av = ctx.structured("claims", f"{iname}: {value}", value, f"{iname}: {value}")
            if av.status == "stated":
                values.append(f"{iname}: {value}")
                eids.extend(av.evidence_ids[:1])
                seen.add(key)
                added = True
    if added:
        out["claims"] = AttributeValue(value=values, status="stated", raw_value=None, evidence_ids=eids)

    # headline plaque / gum claim = the shortest claim naming the topic with a figure
    claims = out["claims"]
    if claims.status != "stated":
        return
    for attribute, topic_rx in (("plaque_removal_claim", re.compile(r"\bplaque\b", re.I)),
                                ("gum_health_claim", re.compile(r"healthier gums|gum health", re.I))):
        picks = [(len(v), i, v) for i, v in enumerate(claims.value)
                 if isinstance(v, str) and len(v) <= 120 and _FIGURE.search(v) and topic_rx.search(v)]
        if not picks:
            continue
        picks.sort()
        _, i, text = picks[0]
        src = ctx.store.get(claims.evidence_ids[i]) if i < len(claims.evidence_ids) else None
        if src is None or src.source_section_index is None or src.source_section == "structured_data":
            continue
        eid = ctx.from_section(attribute, text, text, src.source_section_index)
        if eid:
            ctx.replace(attribute, AttributeValue(value=text, status="stated", raw_value=None, evidence_ids=[eid]))


def _spec_rows(ctx: _Ctx, chapter: str) -> list[tuple[str, list[str], str]]:
    found = []
    if chapter not in ctx.page.chapters:
        return found
    for item in ctx.page.chapters[chapter][0].get("csItem", []):
        iname = str(item.get("csItemName", ""))
        hit = ctx.page.spec_item(re.escape(iname), chapter)
        if hit and hit[1]:
            found.append((iname, hit[1], hit[2]))
    return found


# --- mode_count follows modes ---------------------------------------------------------------------------

def _mode_count(ctx: _Ctx) -> None:
    """mode_count = len(modes) when modes are known. A stated count that
    disagrees is kept, with a warning."""
    modes = ctx.d.get("modes") or []
    detail = ctx.out.get("mode_details")
    if not modes or detail is None or detail.status != "stated":
        return
    n = len(modes)
    current = ctx.out["mode_count"]
    philips_drawer = ctx.page.brand == "Philips" and "Modes" in ctx.page.chapters
    if current.status == "stated" and current.value is not None:
        if current.value != n:
            ctx.warnings.append(NormalizationWarning(
                product_id=ctx.product_id, code="mode_count_disagrees_with_modes", attribute="mode_count",
                message=f"mode_count is stated as {current.value} but {n} modes are listed; the stated count is kept.",
            ))
            return
        if not philips_drawer:
            return
        # same number, but cited from marketing copy: cite the drawer's Modes section instead
    snippet = ctx.store.get(detail.evidence_ids[0]).evidence_text
    # a count of the listed modes (derived), not a sentence from the site
    ctx.replace("mode_count", ctx.structured("mode_count", n, snippet, f"{n} modes listed in the Modes section (derived count)"))


# --- cleaning technology, brush head design (verbatim sentences only) ------------------------------------

_TECH_PRIORITY = [
    r"micro-?vibration", r"oscillat", r"sonic technology", r"sonicare technology",
    r"next-generation sonicare", r"bristle movements|brush movements", r"magnetic (?:iO )?(?:system|technology)",
]


def _best_by_priority(candidates: list[tuple[str, object]], patterns: list[str]):
    """candidates = [(text, payload)]; first candidate matching the earliest pattern."""
    for pattern in patterns:
        rx = re.compile(pattern, re.I)
        for text, payload in candidates:
            if rx.search(fold(text)):
                return text, payload
    return None


def _rank_sentence(text: str, patterns: list[str], faq: bool, own: Optional[re.Pattern] = None) -> tuple:
    """Lower is better: earliest-priority pattern, then a sentence that names
    this very product, then non-FAQ over FAQ (the generic "Oral-B iO
    electric toothbrushes feature ..." answer is the last resort), then more
    matched patterns, then one that also names the brush head, then shorter."""
    folded = fold(text)
    hits = sum(1 for p in patterns if re.search(p, folded, re.I))
    rank = next((i for i, p in enumerate(patterns) if re.search(p, folded, re.I)), len(patterns))
    names_own = 0 if (own and own.search(text)) else 1
    return (rank, names_own, 1 if faq else 0, -hits, 0 if re.search(r"brush head", folded, re.I) else 1, len(text))


# short design phrase: "dentist-inspired round brush head" (never the whole sentence)
_DESIGN_PHRASES = [
    re.compile(r"dentist[-‐-― ]inspired round brush heads?(?: design)?", re.I),
    re.compile(r"(?:unique |small )?round brush heads?", re.I),
    re.compile(r"\bROUND brush heads\b"),
]


def _technology(ctx: _Ctx) -> None:
    """The page's own sentence naming the motion/cleaning technology - never
    inferred from other products. Plus the brush head design as a short phrase
    ("dentist-inspired round brush head"), blank when the page has none."""
    tokens = product_tokens(ctx.record["product_name"], ctx.product_id)
    own = re.compile("|".join(re.escape(t) for t in tokens), re.I) if tokens else None
    for topic, attribute, patterns in (
        ("technology", "cleaning_technology", _TECH_PRIORITY),
        ("brush_head_design", "brush_head_design", [r"round brush head", r"dentist[- ]inspired"]),
    ):
        matches = [re.compile(p, re.I) for p in patterns]
        candidates: list[tuple[tuple, str, object]] = []
        # page sections (description, lead paragraph, FAQ answers) for both brands
        for sent in collect_topic(ctx.sections, topic, limit=40):
            if len(sent.text) <= 400 and any(rx.search(fold(sent.text)) for rx in matches):
                faq = ctx.record["sections"][sent.section_index]["heading"].strip().endswith("?")
                candidates.append((_rank_sentence(sent.text, patterns, faq, own), sent.text, ("section", sent)))
        if ctx.page.brand == "Philips":
            for text, snippet in _philips_topic(ctx, topic, 40):
                if len(text) <= 320 and any(rx.search(fold(text)) for rx in matches):
                    candidates.append((_rank_sentence(text, patterns, False, own), text, ("structured", snippet)))
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0])
        if attribute == "brush_head_design":
            done = False
            for phrase_rx in _DESIGN_PHRASES:
                for _, text, (kind, payload) in candidates:
                    m = phrase_rx.search(text)
                    if m and kind == "section":
                        phrase = m.group(0)
                        eid = ctx.from_section(attribute, phrase, phrase, payload.section_index)
                        if eid:
                            ctx.replace(attribute, AttributeValue(value=phrase, status="stated", raw_value=None, evidence_ids=[eid]))
                            done = True
                            break
                if done:
                    break
            continue
        _, text, (kind, payload) = candidates[0]
        if kind == "section":
            eid = ctx.from_section(attribute, text, text, payload.section_index)
            if eid:
                ctx.replace(attribute, AttributeValue(value=text, status="stated", raw_value=None, evidence_ids=[eid]))
        else:
            ctx.replace(attribute, ctx.structured(attribute, text, payload))


# --- intensity count from page text ---------------------------------------------------------------------

_WORD_NUM = {"two": 2, "three": 3, "four": 4, "five": 5}
_INTENSITY_RX = re.compile(r"\b(\d+|two|three|four|five)\s+intensity\s+(?:settings?|levels?)\b", re.I)


def _intensity_count(ctx: _Ctx) -> None:
    if ctx.out["intensity_count"].status == "stated":
        return
    for idx, section in ctx.sections:
        for line in section["text"].split("\n"):
            m = _INTENSITY_RX.search(fold(line))
            if m and "?" not in line:
                raw = m.group(1).lower()
                n = int(raw) if raw.isdigit() else _WORD_NUM[raw]
                sentence = next((x.strip() for x in split_sentences(line) if m.group(0) in x), m.group(0))
                eid = ctx.from_section("intensity_count", n, sentence, idx, raw_value=m.group(0))
                if eid:
                    ctx.out["intensity_count"] = AttributeValue(value=n, status="stated", raw_value=m.group(0), evidence_ids=[eid])
                    return


# --- replacement reminder: a DEVICE feature, never advice ----------------------------------------------------

_REPLACE_RX = re.compile(r"\brefill change (?:alert|reminder)\b|\b(?:replace\w*|change)\b[^.]{0,40}\b(?:brush|refill)[- ]?(?:head)?s?\b|\b(?:brush|refill) heads?\b[^.]{0,40}\b(?:replace\w*|change)\b", re.I)
_DEVICE_VERB_RX = re.compile(r"\b(?:signals?|alerts?|remind\w*|tracker|tracks?|indicator|monitors?|notif\w+|lets you know)\b", re.I)
_ADVICE_RX = re.compile(r"\bevery\s+(?:\d+|three|3)\s+months?\b|\bor sooner\b|\bwe recommend\b|\bdentists? recommend", re.I)


def _replacement_reminder(ctx: _Ctx) -> None:
    """True only when a sentence says the product itself signals/tracks when
    to replace the head ("...and signal when it's time to replace your brush
    head"). Plain advice ("replace every 3 months or sooner if worn") never
    sets it."""
    current = ctx.out["replacement_reminder"]
    if current.status == "stated" and not (
        current.evidence_ids and len(ctx.store.get(current.evidence_ids[0]).evidence_text) < 30
    ):
        return  # (a bare label like 'REFILL CHANGE ALERT' is upgraded to the sentence that states it)
    for idx, section in ctx.sections:
        if re.search(r"warranty|featured", section["heading"], re.I):
            continue
        for line in section["text"].split("\n"):
            for sentence in split_sentences(line):
                sentence = sentence.strip()
                if len(sentence) < 25 or "?" in sentence:
                    continue
                folded = fold(sentence)
                m = _REPLACE_RX.search(folded)
                if not m:
                    continue
                window = folded[max(0, m.start() - 90): m.end() + 90]
                if not _DEVICE_VERB_RX.search(window):
                    continue
                if _ADVICE_RX.search(folded) and not re.search(r"\b(?:signals?|alerts?|remind\w*|tracker|indicator)\b", window, re.I):
                    continue
                eid = ctx.from_section("replacement_reminder", True, sentence, idx)
                if eid:
                    ctx.out["replacement_reminder"] = AttributeValue(value=True, status="stated", raw_value=None, evidence_ids=[eid])
                    return


# --- Oral-B battery text, verbatim ---------------------------------------------------------------------------------

def _battery_note(ctx: _Ctx) -> None:
    if ctx.page.brand != "Oral-B":
        return
    rx = re.compile(r"charge lasts|\bbattery\b[^.]{0,60}\b\d+\s*(?:days?|weeks?)|\b\d+\s*(?:days?|weeks?)\b[^.]{0,40}\bbattery", re.I)
    charging_time = re.compile(r"\b(?:charging|charge)\b[^.]{0,40}\btakes?\b[^.]{0,20}\d+\s*hours?", re.I)
    for s in collect_topic(ctx.sections, "battery", limit=12):
        # "Charging a low battery takes about 3 hours" is charge time, not runtime
        if rx.search(s.text) and not charging_time.search(s.text):
            eid = ctx.from_section("battery_life_note", s.text, s.text, s.section_index)
            if eid:
                ctx.out["battery_life_note"] = AttributeValue(value=s.text, status="stated", raw_value=None, evidence_ids=[eid])
                return


# --- Philips flags from feature panels + spec rows -----------------------------------------------------------------

_PHILIPS_FLAG_RX = {
    "timer": re.compile(r"timer|pacer", re.I),
    "app_connectivity": re.compile(r"\bapp\b", re.I),
    "bluetooth_connectivity": re.compile(r"bluetooth", re.I),
    "pressure_sensing": re.compile(r"pressure", re.I),
}


def _philips_flags(ctx: _Ctx) -> None:
    """After the generic pass: a flag the marketing sections missed is still
    True when a feature panel or specification row states it. Evidence is that
    exact text (a verbatim sentence, else the spec row's JSON)."""
    page = ctx.page
    features = _philips_features(page)
    for attribute, rx in _PHILIPS_FLAG_RX.items():
        if ctx.out[attribute].status == "stated":
            continue
        hit = None
        for item in features:
            for text in [str(item.get("featureName") or ""), str(item.get("featureLongDescription") or "")] + split_sentences(
                str(item.get("featureGlossary") or "")
            ):
                text = text.strip()
                if len(text) >= 12 and rx.search(text) and text in page.unescaped:
                    hit = (text, text)
                    break
            if hit:
                break
        if hit is None:
            for cname, (obj, _region) in page.chapters.items():
                if cname in ("Modes", "Intensities"):
                    continue
                for item in obj.get("csItem", []):
                    iname = str(item.get("csItemName", ""))
                    values = " ".join(str(v.get("csValueName", "")) for v in item.get("csValue", []))
                    if rx.search(iname) or rx.search(values) or (attribute != "pressure_sensing" and rx.search(cname)):
                        found = page.spec_item(re.escape(iname), cname)
                        if found and found[2]:
                            hit = (f"{cname}: {iname} - {values}".strip(), found[2])
                            break
                if hit:
                    break
        if hit is None and attribute == "timer":
            # pages without a spec row describe the pacing in prose ("The One buzzes
            # every 30 seconds ... A longer buzz at 2 minutes"); quote the rendered text
            m = re.search(r"[^\n.]*\bbuzz\w*\s+every\s+\d+\s+seconds[^\n.]*\.?", page.rendered)
            if m:
                hit = (m.group(0).strip(), m.group(0).strip())
                if ctx.out["timer_description"].status != "stated":
                    ctx.out["timer_description"] = ctx.structured("timer_description", hit[0], hit[1])
        if hit:
            text, snippet = hit
            ctx.replace(attribute, ctx.structured(attribute, True, snippet, text))


def _philips_rendered_fallbacks(ctx: _Ctx) -> None:
    """Facts the rendered page states in plain text but the spec table lacks
    (the One / One For Kids have no spec table rows for them)."""
    r, out = ctx.page.rendered, ctx.out
    if not r:
        return
    if out["warranty_years"].status != "stated":
        m = re.search(r"^(\d+)[\s-]*year warranty\s*$", r, re.I | re.M)
        if m:
            ctx.replace("warranty_years", ctx.structured("warranty_years", int(m.group(1)), m.group(0).strip(), m.group(0).strip()))
    if out["brush_movements_per_min"].status != "stated":
        m = re.search(r"\b(\d{1,3}(?:,\d{3})+)\s*micro-?vibrations?(?:\s+per\s+minute)?", r, re.I)
        if m:
            ctx.replace(
                "brush_movements_per_min",
                ctx.structured("brush_movements_per_min", int(m.group(1).replace(",", "")), m.group(0), m.group(0)),
            )


def _philips_replacement(ctx: _Ctx) -> None:
    """Philips: replacement_reminder is true only when the Replacement
    Reminder spec row carries a visible statement ("... lets you know when to
    replace your brush head"), never because of the row label alone."""
    page, out = ctx.page, ctx.out
    hit = page.spec_item(r"Replacement Reminder")
    old = out["replacement_reminder"]
    if hit and hit[1]:
        text = hit[1][0].strip()
        if re.search(r"replace", text, re.I) and _DEVICE_VERB_RX.search(text):
            new = ctx.structured("replacement_reminder", True, text, f"Replacement Reminder: {text}")
            if new.status == "stated":
                ctx.replace("replacement_reminder", new)
                return
    if old.status == "stated":
        _supersede(ctx.store, old)
        out["replacement_reminder"] = not_mentioned()


# --- availability + price visibility ------------------------------------------------------------

_STATE_LABEL = {"in_stock": "In stock", "out_of_stock": "Out of stock"}


def _visible_availability(page: _Page) -> Optional[tuple[str, str]]:
    """(state, label) read from the visible buy box, or None when the page shows neither cue.

    Oral-B: ``Add to cart ($329.99)`` -> in stock; no add-to-cart but ``Find A Retailer``
    in the buy box -> out of stock. "Out of stock" is our reading of "Find A Retailer";
    the site does not write it, so the label is always stored next to the status.
    Philips: the rendered page offers "Add to cart"."""
    rendered = page.rendered
    if page.brand == "Philips":
        return ("in_stock", "Add to cart") if "Add to cart" in rendered else None
    marker = "=== Buy box ==="
    if marker not in rendered:
        return None
    box = rendered.split(marker, 1)[1]
    m = re.search(r"^.*Add to cart.*$", box, re.I | re.M)
    if m:
        return "in_stock", m.group(0).strip()
    m = re.search(r"^.*Find A Retailer.*$", box, re.I | re.M)
    if m:
        return "out_of_stock", m.group(0).strip()
    return None


def _availability_and_price(ctx: _Ctx) -> None:
    """Availability is cited from visible text. Page data (JSON-LD / store data) is only a
    cross-check: when the rendered page is available and shows neither cue the status is
    not_stated, and a disagreement with the page data is reported as a warning."""
    d, page = ctx.d, ctx.page
    json_state = d.get("availability")
    visible = _visible_availability(page)
    has_snapshot = bool(page.rendered)
    if visible:
        state, raw = visible
        ctx.out["availability"] = ctx.structured("availability", _STATE_LABEL.get(state, state), raw, raw)
        if json_state and json_state != state:
            ctx.warnings.append(NormalizationWarning(
                product_id=ctx.product_id, code="availability_page_data_disagrees", attribute="availability",
                message=f"The visible page shows {raw!r} ({_STATE_LABEL.get(state, state)}) but the page data says {json_state}; the visible page is used.",
            ))
    elif has_snapshot and page.brand == "Oral-B":
        state = None
        if json_state:
            ctx.warnings.append(NormalizationWarning(
                product_id=ctx.product_id, code="availability_not_visible", attribute="availability",
                message=f"The visible buy box shows neither 'Add to cart' nor 'Find A Retailer', so availability is not_stated (page data says {json_state}).",
            ))
    else:
        # no rendered snapshot for this record: page data only, kept as cross-check evidence
        state = json_state
        if state:
            label = _STATE_LABEL.get(state, state)
            if page.brand == "Philips":
                m = re.search(r'"availability":\{"status":"[A-Z_]+"', page.unescaped)
            else:
                m = re.search(r'"availability"\s*:\s*"https://schema\.org/[A-Za-z]+"', page.html)
            if m:
                ctx.out["availability"] = ctx.structured("availability", label, m.group(0), m.group(0))
    state = visible[0] if visible else state
    if state == "out_of_stock" and (d.get("price_displayed") is False or visible):
        hidden = ctx.out["original_price"].value if ctx.out["original_price"].status == "stated" else None
        for name in _PRICE_GROUP:
            if ctx.out[name].status == "stated":
                _supersede(ctx.store, ctx.out[name])
                ctx.out[name] = not_mentioned()
        snippet = _price_snippets(page)[0]
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
            r"(\d+(?:\.\d+)?)[- ]hours?\s+(?:fast\s+)?charg|"
            r"\b(\d+(?:\.\d+)?)H\s+fast\s+charger\b",
            re.I,
        )
        for s in pool:
            m = rx.search(s.text)
            if m:
                value = float(m.group(1) or m.group(2) or m.group(3))
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


_BOTH_FLAG_RX = {
    "bluetooth_connectivity": re.compile(r"bluetooth", re.I),
    "app_connectivity": re.compile(r"\bapp\b", re.I),
    "timer": re.compile(r"\btimer\b", re.I),
    "pressure_sensing": re.compile(r"pressure", re.I),
}


def _both_models_flags(ctx: _Ctx) -> None:
    """"Both models connect with the Oral-B app via Bluetooth" in this
    product's own comparison FAQ is a statement about this product too."""
    tokens = product_tokens(ctx.record["product_name"], ctx.product_id)
    if not tokens:
        return
    number = re.search(r"(\d+)\s*$", tokens[0])  # "iO Series 6" -> 6
    # The heading names this model either in full ("iO Series 6") or as the
    # second number of "Series 5 and 6".
    own = re.compile("|".join(re.escape(t) for t in tokens) + (rf"|\band {number.group(1)}\b" if number else ""), re.I)
    for idx, section in enumerate(ctx.record["sections"]):
        heading = section["heading"]
        if not own.search(heading) or not re.search(r"difference|\bvs\b|versus|compar", heading, re.I):
            continue
        for line in section["text"].split("\n"):
            for sentence in split_sentences(line):
                sentence = sentence.strip()
                if not re.match(r"both (?:models|brushes|products)\b", sentence, re.I):
                    continue
                for attribute, rx in _BOTH_FLAG_RX.items():
                    if ctx.out[attribute].status != "stated" and rx.search(sentence):
                        eid = ctx.from_section(attribute, True, sentence, idx, raw_value="stated for both compared models")
                        if eid:
                            ctx.out[attribute] = AttributeValue(
                                value=True, status="stated", raw_value="stated for both compared models", evidence_ids=[eid])


def _own_side(ctx: _Ctx) -> None:
    if ctx.page.brand != "Oral-B":
        return
    _both_models_flags(ctx)
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

    # specifications = every row of every section in the Technical Specifications drawer (no name list)
    texts, eids = [], []
    for cname in chapters:
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
            # the count is the number of listed levels (never left blank beside the levels)
            ctx.replace(
                "intensity_count",
                ctx.structured("intensity_count", len(spec_names), spec_rows[0][2], f"{len(spec_names)} intensity levels listed",
                               [r[2] for r in spec_rows[1:]]),
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
