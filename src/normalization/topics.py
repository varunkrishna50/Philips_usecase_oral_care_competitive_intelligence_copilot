"""Collect everything a product page says about one topic (pressure,
timer, battery, ...) as verbatim sentences, instead of keeping a single
pick. Two sources:

* ``collect_topic`` - the page's own usable sections (description bullets,
  FAQ answers, tabs). A section whose heading matches the topic (e.g. the
  FAQ "Does the handle vibrate when it is time to switch quadrants?") is
  taken whole; elsewhere only matching sentences are taken.
* ``own_side_facts`` - comparison FAQs ("iO2 vs Pro", "Series 2 vs 3",
  "Series 9 vs 10") are excluded from normal evidence because they mix two
  products. Their lines about the page's *own* product are still facts about
  it; this picks only those clauses (never the other model's).

Every returned sentence is a verbatim substring of its section, so it can be
stored as evidence without further checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .filters import COMPARISON_HEADING_PATTERNS
from .utils import split_sentences

# --- topic definitions --------------------------------------------------------

TOPICS: dict[str, dict[str, str]] = {
    "pressure": {
        "sentence": r"pressure|too hard|too soft|too much force|overbrush|over-brush|brush(?:ing)? too",
        "heading": r"pressure",
    },
    "app": {
        "sentence": r"\bapp\b|coaching|smartphone|\bsync\b|progress tracking",
        "heading": r"\bapp\b|coach|smartphone|sync|apple health",
    },
    "bluetooth": {
        "sentence": r"bluetooth|\bconnects? to\b|\bconnected\b|\bpair(?:ing|ed)?\b",
        "heading": r"bluetooth|\bconnect|\bpair(?:ing)?\b",
    },
    "timer": {
        "sentence": (
            r"\btimer\b|quadrant|sextant|pacer|every 30 seconds|30[- ]second|20[- ]second|"
            r"2[- ]minute|two[- ]minute|clock|session is complete"
        ),
        "heading": r"timer|quadrant|sextant|vibrate|clock|brushing time|switch",
    },
    "battery": {
        "sentence": (
            r"\bbattery\b|charge lasts|low-charge|charging a low|takes about \d+(?:\.\d+)? hours?|"
            r"\d+[- ]hour (?:fast )?charg|full charge|fully charged|\d+ days? of (?:regular )?brushing"
        ),
        "heading": r"battery|how long.{0,25}(?:charge|last)",
    },
    "charging": {
        "sentence": r"\bcharger\b|\bcharging (?:base|station|stand|dock)\b|\bplug(?:ged)?\b|\bdock(?:ed)?\b|\bUSB\b",
        "heading": r"how.{0,15}charge|charger|charging",
    },
    "coverage": {
        "sentence": r"coverage|position detection|tracks? where you brush|teeth tracking|mouth zones?",
        "heading": r"coverage|tracking",
    },
    "display": {
        "sentence": r"\bdisplay\b|light ring|\bscreen\b|handle indicator|visible indicator",
        "heading": r"display|indicator",
    },
}

_BOILERPLATE = re.compile(
    r"consumer health data|by registering|privacy policy|sign[- ]?up|newsletter|"
    r"warranty|money[- ]back guarantee|free shipping|add to cart|find a retailer|"
    r"repair|service center|return shipping|proof of purchase|registering|health conditions|\bcrest\b|p&g|save up to \$|"
    r"you consent|collection of that information|your consumer|processors who|we will use your|disclosure",
    re.I,
)
_SKIP_HEADINGS = re.compile(r"^(?:sort by|page content|reviews?|you may also like|warranty information|featured in)", re.I)


@dataclass
class TopicSentence:
    text: str
    section_index: int
    section_heading: str
    source_kind: str = "page"  # "page" | "comparison_own_side"
    compared_with: Optional[str] = None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _dedupe(items: list[TopicSentence]) -> list[TopicSentence]:
    """Drop repeats and sentences wholly contained in a longer kept one."""
    kept: list[TopicSentence] = []
    for item in items:
        n = _norm(item.text)
        if not n:
            continue
        dup = False
        for i, other in enumerate(kept):
            m = _norm(other.text)
            if n == m or n in m:
                dup = True
                break
            if m in n:
                kept[i] = item
                dup = True
                break
        if not dup:
            kept.append(item)
    return kept


def _sentences_of(section: dict) -> list[str]:
    out: list[str] = []
    for line in section["text"].split("\n"):
        for sentence in split_sentences(line):
            sentence = sentence.strip()
            if sentence:
                out.append(sentence)
    return out


TOPIC_LIMITS = {"timer": 10, "pressure": 8, "battery": 8, "charging": 6}
# Sentences that carry a figure (durations, hours) are the most informative
# for these topics, so they are kept first when the limit bites.
_FIGURES_FIRST = {"timer", "battery"}


def collect_topic(sections, topic: str, limit: Optional[int] = None) -> list[TopicSentence]:
    """``sections`` is the output of ``filters.usable_sections`` ([(index, section)])."""
    limit = limit or TOPIC_LIMITS.get(topic, 6)
    spec = TOPICS[topic]
    sentence_re = re.compile(spec["sentence"], re.I)
    heading_re = re.compile(spec["heading"], re.I)
    description: list[TopicSentence] = []
    faq: list[TopicSentence] = []
    other: list[TopicSentence] = []
    for idx, section in sections:
        heading = section["heading"]
        if _SKIP_HEADINGS.match(heading.strip()):
            continue
        is_question = heading.strip().endswith("?")
        take_whole = bool(is_question and heading_re.search(heading))
        bucket = description if heading == "Product description" else faq if is_question else other
        for sentence in _sentences_of(section):
            if sentence.endswith("?") or len(sentence) < 25 or _BOILERPLATE.search(sentence):
                continue
            if take_whole or sentence_re.search(sentence):
                bucket.append(TopicSentence(sentence, idx, heading))
    # FAQ answers first (they are the most specific), then the description
    # bullets, then anything else on the page.
    ordered = _dedupe(faq + description + other)
    if topic in _FIGURES_FIRST and len(ordered) > limit:
        # When the limit bites keep the sentences with figures, but show
        # them in page order.
        ranked = sorted(range(len(ordered)), key=lambda i: (0 if re.search(r"\d", ordered[i].text) else 1, i))
        keep = set(ranked[:limit])
        ordered = [s for i, s in enumerate(ordered) if i in keep]
    return ordered[:limit]


# --- own-side facts from comparison sections ---------------------------------------

_LABEL_TO_FIELD = [
    (r"motor|cleaning technology|technology", "cleaning_technology"),
    (r"in the box|fewer brush|more brush", "box_note"),
    (r"^brush heads?$", "brush_head_compatibility"),
    (r"feel|noise|quiet", "feel_and_noise"),
    (r"^price", "price_positioning"),
    (r"display|indicator", "display_type"),
    (r"pacing|zones?|intervals?", "pacing"),
    (r"pressure", "pressure_feedback"),
]
_CLAUSE_SPLIT = re.compile(r";\s+|,?\s+(?:while|whereas|compared (?:to|with)|versus|vs\.?)\s+", re.I)
_OTHER_MODEL = re.compile(r"\bSeries \d+|\biO ?\d+\b|\bPro\b|\bGenius\b", re.I)


def product_tokens(product_name: str, product_id: str) -> list[str]:
    """Names this product goes by in comparison copy: 'iO Series 2' -> iO2,
    iO Series 2, Series 2, iO 2 (never matching 'Series 20' or 'Series 2x')."""
    m = re.search(r"(?:io\s*)?series\s*(\d+)", product_name, re.I) or re.search(r"series[- ](\d+)", product_id, re.I)
    if not m:
        return []
    n = m.group(1)
    return [f"iO Series {n}", f"iO{n}", f"iO {n}", f"Series {n}"]


def _token_re(tokens: list[str]) -> Optional[re.Pattern]:
    if not tokens:
        return None
    alts = "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
    return re.compile(rf"(?<![\w]){'(?:' + alts + ')'}(?![\w])", re.I)


def own_side_facts(sections, product_name: str, product_id: str) -> list[dict]:
    """Facts about this product from comparison-framed sections.

    Returns [{field, text, section_index, section_heading, compared_with}].
    Handles both layouts seen on the pages:
      * a label line followed by "<model>: text" rows (Pro vs iO2 FAQ)
      * "Label: sentence about model A, while model B ..." (2 vs 3, 9 vs 10)
    """
    own = _token_re(product_tokens(product_name, product_id))
    if own is None:
        return []
    out: list[dict] = []
    for idx, section in sections:
        heading = section["heading"]
        if not any(p.search(heading) for p in COMPARISON_HEADING_PATTERNS):
            continue
        compared = None
        m = re.search(r"between\s+(?:the\s+)?(.+?)\s+and\s+(.+?)\??$", heading, re.I)
        if m:
            compared = m.group(2) if own.search(m.group(1)) else m.group(1)
        label = ""
        for line in section["text"].split("\n"):
            line = line.strip()
            if not line:
                continue
            # a bare label line: "Motor/cleaning technology" or "Brush heads:"
            if len(line) <= 45 and re.fullmatch(r"[A-Za-z/&' \-]{3,45}:?", line):
                label = line.rstrip(":").strip()
                continue
            head = re.match(r"^(?P<who>[^:]{1,40}):\s*(?P<text>.+)$", line)
            if not head:
                continue
            who, text = head.group("who").strip(), head.group("text").strip()
            if own.fullmatch(who) or own.fullmatch(who.replace("Oral-B ", "")):
                # "<this model>: text" row under the current label
                field = _field_for(label)
                if field and text:
                    out.append(_fact(field, text, idx, heading, compared))
                continue
            if _OTHER_MODEL.fullmatch(who) or len(who) <= 12:
                continue  # a row about another model ("Pro: ...")
            # "Label: sentence ..." - the sentence may name this product first
            field = _field_for(who)
            for clause in _CLAUSE_SPLIT.split(text):
                clause = clause.strip().rstrip(",;")
                if not field or not own.search(clause):
                    continue
                if [m for m in _OTHER_MODEL.finditer(clause) if not own.fullmatch(m.group(0))]:
                    continue
                out.append(_fact(field, clause, idx, heading, compared))
    return _dedupe_facts(out)


def _field_for(label: str) -> Optional[str]:
    for pattern, field in _LABEL_TO_FIELD:
        if re.search(pattern, label.strip(), re.I):
            return field
    return None


def _fact(field: str, text: str, idx: int, heading: str, compared: Optional[str]) -> dict:
    return {"field": field, "text": text, "section_index": idx, "section_heading": heading, "compared_with": compared}


def _dedupe_facts(facts: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out = []
    for f in facts:
        key = (f["field"], _norm(f["text"]))
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
