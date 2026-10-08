"""3C.6: aggregates the SYSTEM'S OWN citation-validation results
(src.rag.citations.ValidatedSegment, now carrying a `trusted` flag --
see src/rag/citations.py) into presence/validity rates.

Explicitly NOT the independent groundedness checker (groundedness.py) --
this module answers "what did the system's own validator decide",
which is near-100% valid by construction for trusted segments (they
skip the validator entirely). Reported separately and labelled as such
in results.json/the report, per the 3C.1 integrity rule.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CitationMetrics:
    total_segments: int
    trusted_segments: int
    untrusted_segments: int
    presence_rate: float  # fraction of segments carrying >=1 citation_id
    validity_rate_all: float  # fraction valid, over ALL segments (inflated by trusted)
    validity_rate_untrusted_only: float  # fraction valid, over untrusted (LLM) segments only


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

    return CitationMetrics(
        total_segments=total,
        trusted_segments=len(trusted),
        untrusted_segments=len(untrusted),
        presence_rate=presence_rate,
        validity_rate_all=validity_rate_all,
        validity_rate_untrusted_only=validity_rate_untrusted_only,
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

    return CitationMetrics(
        total_segments=total, trusted_segments=trusted, untrusted_segments=untrusted,
        presence_rate=presence, validity_rate_all=validity_all, validity_rate_untrusted_only=validity_untrusted,
    )
