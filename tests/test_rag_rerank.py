"""Tests for the deterministic reranker (3A.10): exact product+attribute
beats same-product-other-evidence beats same-brand beats everything
else, and tier always dominates raw hybrid score.
"""
from __future__ import annotations

from src.rag.models import ProductRecord, RetrievalDocument, ScoredDocument
from src.rag.query_understanding import ProductMatch
from src.rag.rerank import rerank


def _doc(product_id, brand, attribute) -> RetrievalDocument:
    return RetrievalDocument(
        document_id=f"{product_id}:{attribute}", evidence_id=f"{product_id}:{attribute}",
        product_id=product_id, brand=brand, product_name=product_id, attribute=attribute,
        value=True, status="stated", evidence_text="x", source_url="https://oralb.com/x",
        source_section="s", source_file="x.html", retrieval_text="x",
    )


def _scored(product_id, brand, attribute, hybrid_score) -> ScoredDocument:
    return ScoredDocument(document=_doc(product_id, brand, attribute), hybrid_score=hybrid_score)


def _product(product_id, brand) -> ProductRecord:
    return ProductRecord(
        product_id=product_id, brand=brand, product_name=product_id, product_url="x",
        discovered_url="x", source_listing_url="x", retrieved_at="2026-01-01T00:00:00Z",
    )


class TestRerank:
    def test_exact_product_and_attribute_wins_even_with_lower_score(self):
        target_match = ProductMatch(kind="matched", candidates=[_product("series10", "Oral-B")])
        docs = [
            _scored("series9", "Oral-B", "pressure_sensing", hybrid_score=0.99),  # other product, high score
            _scored("series10", "Oral-B", "pressure_sensing", hybrid_score=0.01),  # exact match, low score
        ]
        result = rerank(docs, target_match, ["pressure_sensing"])
        assert result[0].document.product_id == "series10"
        assert result[0].rerank_tier == 1

    def test_same_product_other_attribute_beats_same_brand_other_product(self):
        target_match = ProductMatch(kind="matched", candidates=[_product("series10", "Oral-B")])
        docs = [
            _scored("series9", "Oral-B", "pressure_sensing", hybrid_score=0.9),   # same brand, other product
            _scored("series10", "Oral-B", "timer", hybrid_score=0.1),             # same product, other attribute
        ]
        result = rerank(docs, target_match, ["pressure_sensing"])
        assert result[0].document.product_id == "series10"
        assert result[0].rerank_tier == 2
        assert result[1].rerank_tier == 3

    def test_same_brand_beats_other_brand(self):
        target_match = ProductMatch(kind="matched", candidates=[_product("series10", "Oral-B")])
        docs = [
            _scored("hx9990-13", "Philips", "pressure_sensing", hybrid_score=0.9),
            _scored("series9", "Oral-B", "pressure_sensing", hybrid_score=0.1),
        ]
        result = rerank(docs, target_match, ["pressure_sensing"])
        assert result[0].document.brand == "Oral-B"
        assert result[0].rerank_tier == 3
        assert result[1].rerank_tier == 4

    def test_no_product_requested_falls_back_to_pure_hybrid_score_order(self):
        no_match = ProductMatch(kind="none", candidates=[])
        docs = [
            _scored("series9", "Oral-B", "timer", hybrid_score=0.2),
            _scored("series10", "Oral-B", "timer", hybrid_score=0.8),
        ]
        result = rerank(docs, no_match, ["timer"])
        assert [sd.document.product_id for sd in result] == ["series10", "series9"]
        assert all(sd.rerank_tier == 4 for sd in result)

    def test_semantic_similarity_never_outranks_product_identity(self):
        """A semantically-close but WRONG product, even with a near-
        perfect hybrid score, must never outrank the exact requested
        product (3A.10's core requirement)."""
        target_match = ProductMatch(kind="matched", candidates=[_product("series10", "Oral-B")])
        docs = [
            _scored("series9", "Oral-B", "mode_count", hybrid_score=1.0),
            _scored("series10", "Oral-B", "mode_count", hybrid_score=0.0001),
        ]
        result = rerank(docs, target_match, ["mode_count"])
        assert result[0].document.product_id == "series10"
