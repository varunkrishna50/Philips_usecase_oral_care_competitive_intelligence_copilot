"""Deterministic reranking (3A.10).

Priority tiers, in order:
  1. exact requested product + exact requested attribute
  2. same product, other (still relevant) evidence
  3. same brand, no product match (semantic match only)
  4. everything else (other product, or no product was requested at all)

Tier always dominates the hybrid score -- a tier-1 candidate outranks
every tier-2+ candidate regardless of score, so semantic similarity can
never outrank product identity. Within a tier, hybrid score breaks ties.
No cross-encoder: a simple deterministic tier + score sort is sufficient
for this corpus size, per the 3A.10 spec.
"""
from __future__ import annotations

from .models import ScoredDocument
from .query_understanding import ProductMatch


def _tier(document, product_match: ProductMatch, attributes: list[str]) -> int:
    if product_match.kind != "matched":
        return 4
    requested_product_id = product_match.candidates[0].product_id
    requested_brand = product_match.candidates[0].brand

    product_is_target = document.product_id == requested_product_id
    attribute_is_target = bool(attributes) and document.attribute in attributes

    if product_is_target and attribute_is_target:
        return 1
    if product_is_target:
        return 2
    if document.brand == requested_brand:
        return 3
    return 4


def rerank(
    scored_docs: list[ScoredDocument],
    product_match: ProductMatch,
    attributes: list[str],
) -> list[ScoredDocument]:
    for sd in scored_docs:
        sd.rerank_tier = _tier(sd.document, product_match, attributes)
        sd.rerank_score = sd.hybrid_score
    scored_docs.sort(key=lambda sd: (sd.rerank_tier, -sd.rerank_score))
    return scored_docs
