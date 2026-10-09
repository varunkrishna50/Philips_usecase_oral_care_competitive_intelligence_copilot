"""Answer-quality metrics: is an answer concise, free of repeats and readable?

Three measurements, each computed from what the user is shown (the answer text, the Sources
section and the validated segments) so they read the same before and after any change to how
answers are assembled:

* duplicate sources  - Sources entries whose quote repeats an earlier entry for the same product.
* redundant segments - a shown segment whose cited quotes were all cited by earlier segments and
  which adds no number or distinctive word the earlier segments lack.
* jargon             - internal vocabulary in OUR OWN wording ("normalized value", "the available
  source", "evidence", "corpus", ...). Page text is not ours: spans inside quotation marks, the
  ``Quote:`` lines of the Sources section and verbatim evidence sentences are removed first, because a
  real site sentence may legitimately contain "status" or "record".

Pure functions; no dependency on how the pipeline builds its answers.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "has", "have", "its", "also", "are", "was", "you", "your",
    "from", "than", "more", "can", "will", "not", "does", "into", "when", "which", "their", "about", "page",
}

JARGON_RE = re.compile(
    r"\bnormali[sz]ed\b|\bevidence\b|\battributes?\b|\bnull\b|\bstatus\b|\bcorpus\b|\brecords?\b"
    r"|\bthe available sources?\b|\bmanufacturer states\b",
    re.I,
)
_QUOTED_SPAN_RE = re.compile(r"“[^”]*”|\"[^\"]*\"")
_QUOTE_LINE_RE = re.compile(r"^\s*Quote:.*$", re.M)
_SOURCE_HEADER_RE = re.compile(r"^\[S(\d+)\]\s+(.*?)\s+--\s+official product page\s*$", re.M)


def normalize_quote(text: str) -> str:
    text = text.translate({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2011: "-", 0x2013: "-", 0x2014: "-", 0xA0: " "})
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# --- duplicate sources -------------------------------------------------------------------------------------

def parse_sources(sources_text: str) -> list[dict[str, str]]:
    """[{marker, product, quote}] read back from the rendered Sources section."""
    entries = []
    headers = list(_SOURCE_HEADER_RE.finditer(sources_text or ""))
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(sources_text)
        block = sources_text[m.end():end]
        q = re.search(r'^\s*Quote:\s*"(.*)"\s*$', block, re.M)
        entries.append({"marker": f"S{m.group(1)}", "product": m.group(2).strip(), "quote": q.group(1) if q else ""})
    return entries


def duplicate_sources(sources_text: str) -> dict[str, int]:
    """{shown, duplicates}: a Sources entry is a duplicate when an earlier entry has the same product and quote."""
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    entries = parse_sources(sources_text)
    for e in entries:
        key = (e["product"], normalize_quote(e["quote"]))
        if e["quote"] and key in seen:
            duplicates += 1
        seen.add(key)
    return {"shown": len(entries), "duplicates": duplicates}


# --- redundant segments ------------------------------------------------------------------------------------

def _content_tokens(text: str) -> set[str]:
    tokens = {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}
    tokens |= {n.replace(",", "") for n in _NUMBER_RE.findall(text)}
    return tokens


def redundant_segments(segments: Iterable[Any], evidence_by_id: dict[str, Any]) -> dict[str, int]:
    """{shown, redundant} over the valid, cited segments in answer order."""
    seen_keys: set[tuple[str, str]] = set()
    seen_tokens: set[str] = set()
    shown = redundant = 0
    for seg in segments:
        if not getattr(seg, "valid", True) or not getattr(seg, "citation_ids", None):
            continue
        shown += 1
        keys = set()
        for cid in seg.citation_ids:
            doc = evidence_by_id.get(cid)
            if doc is not None:
                keys.add((doc.product_id, normalize_quote(getattr(doc, "shown_text", None) or doc.evidence_text)))
        tokens = _content_tokens(seg.text)
        if keys and keys <= seen_keys and not (tokens - seen_tokens):
            redundant += 1
        seen_keys |= keys
        seen_tokens |= tokens
    return {"shown": shown, "redundant": redundant}


# --- jargon --------------------------------------------------------------------------------------------------

def our_own_wording(answer_text: str, segments: Iterable[Any] = (), evidence_by_id: Optional[dict[str, Any]] = None) -> str:
    """The answer with page text removed: quotation-mark spans, ``Quote:`` lines and any cited
    evidence sentence that appears verbatim."""
    text = _QUOTE_LINE_RE.sub("", answer_text or "")
    text = _QUOTED_SPAN_RE.sub(" ", text)
    for seg in segments or ():
        for cid in getattr(seg, "citation_ids", None) or ():
            doc = (evidence_by_id or {}).get(cid)
            if doc is None:
                continue
            for sentence in {doc.evidence_text, getattr(doc, "shown_text", "")}:
                if sentence and len(sentence) > 8:
                    text = text.replace(sentence, " ")
    return text


def jargon_hits(answer_text: str, segments: Iterable[Any] = (), evidence_by_id: Optional[dict[str, Any]] = None) -> list[str]:
    return sorted({m.group(0).lower() for m in JARGON_RE.finditer(our_own_wording(answer_text, segments, evidence_by_id))})


# --- per answer / aggregate -----------------------------------------------------------------------------------

def measure_answer(answer, evidence_by_id: dict[str, Any]) -> dict[str, Any]:
    """Quality numbers for one AnswerResult (anything with answer_text, sources_text, segments)."""
    src = duplicate_sources(answer.sources_text)
    red = redundant_segments(answer.segments, evidence_by_id)
    hits = jargon_hits(answer.answer_text, answer.segments, evidence_by_id)
    return {
        "sources_shown": src["shown"], "duplicate_sources": src["duplicates"],
        "segments_shown": red["shown"], "redundant_segments": red["redundant"],
        "jargon_terms": hits,
    }


def aggregate(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [c for c in per_case if c]
    sources = sum(c["sources_shown"] for c in cases)
    segments = sum(c["segments_shown"] for c in cases)
    dup = sum(c["duplicate_sources"] for c in cases)
    red = sum(c["redundant_segments"] for c in cases)
    jargon_cases = [c for c in cases if c["jargon_terms"]]
    return {
        "cases": len(cases),
        "sources_shown": sources, "duplicate_sources": dup,
        "duplicate_source_rate": (dup / sources) if sources else 0.0,
        "segments_shown": segments, "redundant_segments": red,
        "redundant_segment_rate": (red / segments) if segments else 0.0,
        "cases_with_jargon": len(jargon_cases),
        "jargon_rate": (len(jargon_cases) / len(cases)) if cases else 0.0,
    }


def template_samples() -> dict[str, str]:
    """Every user-facing sentence our own code writes (templates, refusals, clarifications), by name.
    Built by calling the real functions/constants, so a reworded or new template is picked up."""
    from src.rag import scope
    from src.rag.attribute_status import AttributeStatus, status_aware_statement
    from src.rag.models import ProductRecord

    product = ProductRecord("p_1", "Philips", "Sonicare Test 4100", "https://example.test", "https://example.test", "https://example.test", "2026-01-01")
    docs = [type("D", (), {"evidence_text": "It lasts a long time.", "shown_text": "It lasts a long time."})()]
    samples: dict[str, str] = {}
    for status, value in (("not_mentioned", None), ("stated_absent", None), ("ambiguous", None), ("stated", None), ("stated", True)):
        for attribute in ("battery_life_days", "pressure_sensing"):
            st = AttributeStatus(product=product, attribute=attribute, status=status, value=value, evidence=docs if status == "stated" else [])
            samples[f"status:{status}:{value}:{attribute}"] = status_aware_statement(st)
    samples["scope:third_party"] = scope.detect_out_of_scope_source("price on amazon") or ""
    samples["scope:non_us"] = scope.detect_out_of_scope_source("the UK page") or ""
    samples["scope:medical"] = scope.MEDICAL_ADVICE_MESSAGE
    samples["scope:no_scope"] = scope.NO_SCOPE_MESSAGE
    from src.rag import retrieve, generation

    for name in ("NOT_IN_CORPUS_MESSAGE", "EMPTY_ANSWER_MESSAGE"):
        for module in (retrieve, generation):
            if hasattr(module, name):
                samples[f"{module.__name__.rsplit('.', 1)[-1]}:{name}"] = getattr(module, name)
    return samples


def template_jargon(samples: Optional[dict[str, str]] = None) -> dict[str, list[str]]:
    samples = samples if samples is not None else template_samples()
    out = {}
    for name, text in samples.items():
        hits = jargon_hits(text)
        if hits:
            out[name] = hits
    return out


BASELINE_PATH = "data/evaluation/baseline_answer_quality.json"


def load_baseline(mode: str, path: str = BASELINE_PATH) -> Optional[dict[str, Any]]:
    """The saved 'before' numbers for ``mode`` ("offline" / "live"), or None if there is no baseline file."""
    import json
    from pathlib import Path

    file = Path(__file__).resolve().parents[2] / path
    if not file.exists():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    return data.get(mode)
