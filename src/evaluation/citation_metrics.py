"""3C.6: aggregates the SYSTEM'S OWN citation-validation results
(src.rag.citations.ValidatedSegment, now carrying a `trusted` flag --
see src/rag/citations.py) into presence/validity rates.

Explicitly NOT the independent groundedness checker (groundedness.py) --
this module answers "what did the system's own validator decide",
which is near-100% valid by construction for trusted segments (they
skip the validator entirely). Reported separately and labelled as such
in results.json/the report, per the 3C.1 integrity rule.

Segments without a citation are split by kind, because they are not equally serious:

* ``absence_disclosure``  - a code-generated "The page does not mention X" /
  "The page says X does not have Y" sentence. It states an absence, so no evidence record
  exists that it could cite. Not a factual claim.
* ``rejected_by_validator`` - an LLM segment the citation validator rejected (its citation
  ids are cleared on rejection). It is not shown in the answer.
* ``uncited_other`` - anything else: a factual segment that is shown but has no citation.
  This is a real gap and is gated at zero.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ABSENCE_RE = re.compile(
    r"^\s*(?:the page does not (?:mention|say enough)|the page says .{1,200}? does not have|"
    r"the page gives conflicting information|"
    # earlier wording, kept so saved results from before the rewrite still classify the same way
    r"the available sources? does not (?:mention|contain)|the source explicitly indicates that|"
    r"the available sources contain conflicting)",
    re.I,
)


@dataclass
class CitationMetrics:
    total_segments: int
    trusted_segments: int
    untrusted_segments: int
    presence_rate: float  # fraction of segments carrying >=1 citation_id
    validity_rate_all: float  # fraction valid, over ALL segments (inflated by trusted)
    validity_rate_untrusted_only: float  # fraction valid, over untrusted (LLM) segments only
    # --- uncited segments, by kind (counts add up to total_segments - cited) ---
    uncited_absence_disclosures: int = 0
    uncited_rejected_by_validator: int = 0
    uncited_other: int = 0
    # presence over the factual segments the user actually sees (disclosures and rejected ones excluded)
    presence_rate_shown_factual: float = 1.0
    shown_factual_segments: int = 0


def classify_uncited(segment) -> str:
    if segment.citation_ids:
        return "cited"
    if segment.trusted and _ABSENCE_RE.match(segment.text or ""):
        return "absence_disclosure"
    if not segment.trusted and not segment.valid:
        return "rejected_by_validator"
    return "uncited_other"


def compute_citation_metrics(validated_segments: list) -> CitationMetrics:
    total = len(validated_segments)
    if total == 0:
        return CitationMetrics(0, 0, 0, 0.0, 0.0, 0.0)

    trusted = [s for s in validated_segments if s.trusted]
    untrusted = [s for s in validated_segments if not s.trusted]

    presence_rate = sum(1 for s in validated_segments if s.citation_ids) / total
    validity_rate_all = sum(1 for s in validated_segments if s.valid) / total
    validity_rate_untrusted_only = (
        sum(1 for s in untrusted if s.valid) / len(untrusted) if untrusted else 1.0
    )

    kinds = [classify_uncited(s) for s in validated_segments]
    shown_factual = [
        s for s, k in zip(validated_segments, kinds)
        if k in ("cited", "uncited_other") and s.valid
    ]
    presence_shown = (sum(1 for s in shown_factual if s.citation_ids) / len(shown_factual)) if shown_factual else 1.0

    return CitationMetrics(
        total_segments=total,
        trusted_segments=len(trusted),
        untrusted_segments=len(untrusted),
        presence_rate=presence_rate,
        validity_rate_all=validity_rate_all,
        validity_rate_untrusted_only=validity_rate_untrusted_only,
        uncited_absence_disclosures=kinds.count("absence_disclosure"),
        uncited_rejected_by_validator=kinds.count("rejected_by_validator"),
        uncited_other=kinds.count("uncited_other"),
        presence_rate_shown_factual=presence_shown,
        shown_factual_segments=len(shown_factual),
    )


def aggregate_citation_metrics(per_case: list[CitationMetrics]) -> CitationMetrics:
    total = sum(m.total_segments for m in per_case)
    trusted = sum(m.trusted_segments for m in per_case)
    untrusted = sum(m.untrusted_segments for m in per_case)
    if total == 0:
        return CitationMetrics(0, 0, 0, 0.0, 0.0, 0.0)

    presence = sum(m.presence_rate * m.total_segments for m in per_case) / total
    validity_all = sum(m.validity_rate_all * m.total_segments for m in per_case) / total
    untrusted_valid_total = sum(m.validity_rate_untrusted_only * m.untrusted_segments for m in per_case if m.untrusted_segments)
    validity_untrusted = untrusted_valid_total / untrusted if untrusted else 1.0
    shown = sum(m.shown_factual_segments for m in per_case)
    presence_shown = (sum(m.presence_rate_shown_factual * m.shown_factual_segments for m in per_case) / shown) if shown else 1.0

    return CitationMetrics(
        total_segments=total, trusted_segments=trusted, untrusted_segments=untrusted,
        presence_rate=presence, validity_rate_all=validity_all, validity_rate_untrusted_only=validity_untrusted,
        uncited_absence_disclosures=sum(m.uncited_absence_disclosures for m in per_case),
        uncited_rejected_by_validator=sum(m.uncited_rejected_by_validator for m in per_case),
        uncited_other=sum(m.uncited_other for m in per_case),
        presence_rate_shown_factual=presence_shown, shown_factual_segments=shown,
    )
