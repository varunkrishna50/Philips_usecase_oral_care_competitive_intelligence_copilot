"""How the evidence behind an answer is prepared and trimmed, so an answer states each fact once,
cites it once and uses only what answers the question.

* ``quote_key`` / ``prepare_evidence`` - one record per (product, normalised quote). Step 2 can hold the
  same sentence under several attributes (battery_life_days, battery_details, battery_life_raw); the
  reader should see it once.
* ``limit_related`` - related detail records are kept when the question names them, plus at most the
  best one that scored close to the top retrieval hit; the rest are facts nobody asked for.
* ``cap_citations`` - at most two citations per segment, direct records first, but never fewer than
  it takes to back every figure the segment states.
* ``drop_redundant_segments`` - a later segment that cites only quotes already cited and adds no number
  or distinctive word is dropped.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from .attribute_status import RELATED_ATTRIBUTES
from .citations import Segment, ValidatedSegment, _figures
from .models import RetrievalDocument

MAX_CITATIONS_PER_SEGMENT = 2
# A "mixed" related record is kept only if its retrieval score is at least this fraction of the best hit
# (measured on the iO10 battery question: 0.8 let "charging takes about 3 hours" in, 0.95 does not)
RELATED_MIN_SCORE_RATIO = 0.95
MAX_UNNAMED_RELATED = 1

# Related attributes whose records are all about the asked topic (how the pressure sensor works, what the
# modes are, the list/current price): always kept. The other related attributes (battery_details also
# talks about charging, the travel case and the display; availability and promotions are not the price)
# are "mixed" and limited by limit_related.
TOPIC_PURE_RELATED = {
    "pressure_sensing_details", "app_connectivity_details", "bluetooth_details", "coverage_tracking_details",
    "mode_details", "intensity_details", "timer_description", "brush_head_design",
    "original_price", "price_after_discount",
    # claims reach an answer only through the plaque / gum headline questions, already filtered to that topic;
    # a page can state different figures for the same topic and the answer must show all of them
    "claims",
}

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "has", "have", "its", "also", "are", "was", "you", "your",
    "from", "than", "more", "can", "will", "not", "does", "into", "when", "which", "their", "about", "page",
}
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def normalize_quote(text: str) -> str:
    text = text.translate({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2011: "-", 0x2013: "-", 0x2014: "-", 0xA0: " "})
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_key(doc: RetrievalDocument) -> tuple[str, str]:
    return (doc.product_id, normalize_quote(doc.shown_text))


def directness(doc: RetrievalDocument, asked_attributes: Iterable[str]) -> tuple[int, int, int]:
    """Lower is more direct: the asked attribute's own record, then its detail records, then the rest.
    Within a group a detail record ('*_details', 'timer_description') beats a raw/label one; among the
    asked attributes the one named first wins."""
    asked = list(asked_attributes)
    order = asked.index(doc.attribute) if doc.attribute in asked else 0
    if doc.attribute in asked:
        group = 0
    elif any(doc.attribute in RELATED_ATTRIBUTES.get(a, ()) for a in asked):
        group = 1
    else:
        group = 2
    detail = 0 if (doc.attribute.endswith("_details") or doc.attribute == "timer_description") else 1
    return (group, detail, order)


def prepare_evidence(
    docs: list[RetrievalDocument],
    asked_attributes: list[str],
    already_quoted: Iterable[tuple[str, str]] = (),
) -> tuple[list[RetrievalDocument], dict[str, str]]:
    """(kept records in their original order, {dropped evidence id: kept evidence id}).

    One record per (product, normalised quote), the most direct one. Records whose quote a trusted
    template sentence already shows (``already_quoted``) are dropped; they alias to nothing."""
    covered = set(already_quoted)
    best: dict[tuple[str, str], RetrievalDocument] = {}
    for doc in docs:
        key = quote_key(doc)
        if key in covered:
            continue
        if key not in best or directness(doc, asked_attributes) < directness(best[key], asked_attributes):
            best[key] = doc
    kept_ids = {d.evidence_id for d in best.values()}
    aliases = {}
    for doc in docs:
        key = quote_key(doc)
        if key in best and doc.evidence_id not in kept_ids:
            aliases[doc.evidence_id] = best[key].evidence_id
    kept = [d for d in docs if d.evidence_id in kept_ids]
    seen: set[str] = set()
    ordered = []
    for d in kept:  # keep first occurrence order, unique ids
        if d.evidence_id not in seen:
            seen.add(d.evidence_id)
            ordered.append(d)
    return ordered, aliases


def limit_related(
    related: list[RetrievalDocument],
    named_attrs: set[str],
    scores: dict[str, float],
    min_ratio: float = RELATED_MIN_SCORE_RATIO,
    max_unnamed: int = MAX_UNNAMED_RELATED,
) -> list[RetrievalDocument]:
    """Related detail records worth showing, given ``named_attrs`` (the
    attributes the query was already classified as asking about --
    retrieval_result.attributes -- passed in rather than re-detected
    here, so this never issues its own separate query-classification
    call).

    Kept: every record of a topic-pure related attribute; every record whose attribute the question itself
    names ("charging" names charging_details); and at most ``max_unnamed`` of the remaining mixed records,
    the best by retrieval score, only if that score is within ``min_ratio`` of the top score among the
    query's hits. Everything else is a fact nobody asked for."""
    if not related:
        return []
    kept = [d for d in related if d.attribute in named_attrs or d.attribute in TOPIC_PURE_RELATED]
    rest = [d for d in related if d not in kept]
    top = max(scores.values(), default=0.0)
    ranked = sorted(rest, key=lambda d: -scores.get(d.evidence_id, 0.0))
    for doc in ranked[:max_unnamed]:
        score = scores.get(doc.evidence_id, 0.0)
        if top > 0 and score >= min_ratio * top:
            kept.append(doc)
    order = {d.evidence_id: i for i, d in enumerate(related)}
    return sorted(kept, key=lambda d: order[d.evidence_id])


def cap_citations(
    segment: Segment,
    docs_by_id: dict[str, RetrievalDocument],
    asked_attributes: list[str],
    max_citations: int = MAX_CITATIONS_PER_SEGMENT,
) -> Segment:
    """Keep the ``max_citations`` most direct citations, plus whatever else is needed so every figure
    in the segment's text is still backed by a cited quote."""
    ids = [i for i in dict.fromkeys(segment.citation_ids) if i in docs_by_id]
    if len(ids) <= max_citations:
        return Segment(text=segment.text, citation_ids=ids or list(segment.citation_ids), trusted=segment.trusted)
    ranked = sorted(ids, key=lambda i: (directness(docs_by_id[i], asked_attributes), ids.index(i)))
    keep = ranked[:max_citations]
    name_numbers = set().union(*[_figures(docs_by_id[i].product_name) for i in ids])
    needed = _figures(segment.text) - name_numbers
    covered = set().union(*[_figures(docs_by_id[i].evidence_text) | _figures(docs_by_id[i].shown_text) for i in keep])
    for figure in sorted(needed - covered):
        for i in ranked:
            if i not in keep and figure in (_figures(docs_by_id[i].evidence_text) | _figures(docs_by_id[i].shown_text)):
                keep.append(i)
                covered |= _figures(docs_by_id[i].evidence_text) | _figures(docs_by_id[i].shown_text)
                break
    keep_set = set(keep)
    return Segment(text=segment.text, citation_ids=[i for i in ids if i in keep_set], trusted=segment.trusted)


def _content_tokens(text: str) -> set[str]:
    tokens = {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}
    tokens |= {n.replace(",", "") for n in _NUMBER_RE.findall(text)}
    return tokens


def drop_redundant_segments(
    segments: list[ValidatedSegment], docs_by_id: dict[str, RetrievalDocument],
) -> list[ValidatedSegment]:
    """Drop a valid, untrusted segment that cites only quotes earlier shown segments already cited and adds
    no number or distinctive word. Trusted (code-written) segments and invalid ones are never touched here."""
    seen_keys: set[tuple[str, str]] = set()
    seen_tokens: set[str] = set()
    out: list[ValidatedSegment] = []
    for seg in segments:
        if not seg.valid or not seg.citation_ids:
            out.append(seg)
            continue
        keys = {quote_key(docs_by_id[c]) for c in seg.citation_ids if c in docs_by_id}
        tokens = _content_tokens(seg.text)
        if not seg.trusted and keys and keys <= seen_keys and not (tokens - seen_tokens):
            continue
        seen_keys |= keys
        seen_tokens |= tokens
        out.append(seg)
    return out
