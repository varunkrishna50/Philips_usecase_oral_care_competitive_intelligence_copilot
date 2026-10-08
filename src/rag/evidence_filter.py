"""Hard evidence filtering (3A.11).

Most of this is already structurally guaranteed by Step 2, not
re-implemented here:

- Contamination (reviews, cross-sell, navigation, boilerplate, unrelated
  comparison FAQs): Step 2's filters.usable_sections() excludes these
  BEFORE any evidence is ever created, so no contaminated record can
  exist in evidence.json in the first place.
- Evidence validity (verbatim substring, correct raw file/section):
  Step 2's own validation.py gates evidence.json at build time, and
  scripts/build_table.py refuses to report success if that check fails.
- status != not_mentioned: by construction, Step 2 never creates an
  evidence record for a not_mentioned attribute (see models.py's
  not_mentioned() factory), so no such record can appear here either.

What this module actually adds at query time:
- the source-domain allowlist (defense in depth -- redundant with Step 1
  only ever having scraped the two approved domains, but checked
  explicitly here since the spec calls for it as a retrieval-time gate);
- the HARD product-match filter (if the query named a specific product,
  evidence from any OTHER product is rejected outright, not just
  down-ranked -- reranking is a soft preference, this is a hard gate);
- the HARD attribute filter (if the query named specific attribute(s),
  evidence for any other attribute is rejected).
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from .config import APPROVED_SOURCE_DOMAINS
from .models import ScoredDocument
from .query_understanding import ProductMatch


@dataclass
class FilterResult:
    accepted: list[ScoredDocument]
    rejected: list[tuple[ScoredDocument, str]]


def filter_evidence(
    scored_docs: list[ScoredDocument],
    product_match: ProductMatch,
    attributes: list[str],
) -> FilterResult:
    accepted: list[ScoredDocument] = []
    rejected: list[tuple[ScoredDocument, str]] = []

    requested_product_ids = (
        {p.product_id for p in product_match.candidates} if product_match.kind == "matched" else None
    )

    for sd in scored_docs:
        doc = sd.document
        domain = urlparse(doc.source_url).netloc
        if domain not in APPROVED_SOURCE_DOMAINS:
            rejected.append((sd, f"source domain {domain!r} is not an approved Philips US/Oral-B US host"))
            continue
        if doc.status == "not_mentioned":
            # Should be unreachable (Step 2 never creates such a record),
            # kept as an explicit, documented defensive check.
            rejected.append((sd, "status is not_mentioned; not_mentioned is never evidence"))
            continue
        if requested_product_ids is not None and doc.product_id not in requested_product_ids:
            rejected.append((sd, f"evidence belongs to {doc.product_id!r}, not the requested product"))
            continue
        if attributes and doc.attribute not in attributes:
            rejected.append((sd, f"evidence attribute {doc.attribute!r} not in requested attribute(s) {attributes}"))
            continue
        accepted.append(sd)

    return FilterResult(accepted=accepted, rejected=rejected)
