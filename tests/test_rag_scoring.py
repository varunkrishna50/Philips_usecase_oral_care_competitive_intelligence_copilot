"""Tests for hybrid score normalization/combination (3A.7)."""
from __future__ import annotations

from src.rag.models import RetrievalDocument
from src.rag.scoring import combine_hybrid, min_max_normalize


def _doc(doc_id: str) -> RetrievalDocument:
    return RetrievalDocument(
        document_id=doc_id, evidence_id=doc_id, product_id="p1", brand="Oral-B",
        product_name="p1", attribute="pressure_sensing", value=True, status="stated",
        evidence_text="x", source_url="https://oralb.com/x", source_section="s",
        source_file="x.html", retrieval_text="x",
    )


class TestMinMaxNormalize:
    def test_normalizes_to_zero_one_range(self):
        result = min_max_normalize({"a": 1.0, "b": 5.0, "c": 3.0})
        assert result["a"] == 0.0
        assert result["b"] == 1.0
        assert 0.0 < result["c"] < 1.0

    def test_empty_input_returns_empty(self):
        assert min_max_normalize({}) == {}

    def test_all_equal_nonzero_scores_normalize_to_one(self):
        result = min_max_normalize({"a": 2.0, "b": 2.0})
        assert result == {"a": 1.0, "b": 1.0}

    def test_all_zero_scores_normalize_to_zero(self):
        result = min_max_normalize({"a": 0.0, "b": 0.0})
        assert result == {"a": 0.0, "b": 0.0}


class TestCombineHybrid:
    def test_alpha_one_is_bm25_only(self):
        d1, d2 = _doc("d1"), _doc("d2")
        bm25 = [(d1, 10.0), (d2, 1.0)]
        semantic = [(d1, 0.1), (d2, 0.9)]
        scored = combine_hybrid(bm25, semantic, alpha=1.0)
        assert scored[0].document.document_id == "d1"  # bm25 winner, semantic ignored

    def test_alpha_zero_is_semantic_only(self):
        d1, d2 = _doc("d1"), _doc("d2")
        bm25 = [(d1, 10.0), (d2, 1.0)]
        semantic = [(d1, 0.1), (d2, 0.9)]
        scored = combine_hybrid(bm25, semantic, alpha=0.0)
        assert scored[0].document.document_id == "d2"  # semantic winner, bm25 ignored

    def test_document_present_in_only_one_source_still_gets_a_score(self):
        d1, d2 = _doc("d1"), _doc("d2")
        bm25 = [(d1, 10.0)]
        semantic = [(d2, 0.9)]
        scored = combine_hybrid(bm25, semantic, alpha=0.5)
        ids = {sd.document.document_id for sd in scored}
        assert ids == {"d1", "d2"}

    def test_results_sorted_descending_by_hybrid_score(self):
        d1, d2, d3 = _doc("d1"), _doc("d2"), _doc("d3")
        bm25 = [(d1, 10.0), (d2, 5.0), (d3, 1.0)]
        semantic = [(d1, 0.5), (d2, 0.5), (d3, 0.5)]
        scored = combine_hybrid(bm25, semantic, alpha=0.5)
        scores = [sd.hybrid_score for sd in scored]
        assert scores == sorted(scores, reverse=True)
