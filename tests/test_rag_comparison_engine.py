"""Unit tests for the deterministic comparison engine (3B.1): all
arithmetic/boolean comparisons must be computed in code, never asked of
an LLM, and must refuse to compare when either side isn't cleanly
`stated`."""
from __future__ import annotations

from src.rag.comparison_engine import compare_attribute
from src.rag.models import ProductRecord, RetrievalDocument


def _product(product_id, name, attributes) -> ProductRecord:
    return ProductRecord(
        product_id=product_id, brand="Philips" if "philips" in product_id else "Oral-B",
        product_name=name, product_url="https://example.com", discovered_url="https://example.com",
        source_listing_url="https://example.com", retrieved_at="2026-01-01T00:00:00Z",
        attributes=attributes,
    )


def _doc(product_id, product_name, attribute, status, value, text, evidence_id) -> RetrievalDocument:
    return RetrievalDocument(
        document_id=evidence_id, evidence_id=evidence_id, product_id=product_id,
        brand="Philips" if "philips" in product_id else "Oral-B", product_name=product_name,
        attribute=attribute, value=value, status=status, evidence_text=text,
        source_url="https://example.com", source_section="Features", source_file="t.html",
        retrieval_text=text,
    )


class TestNumericComparison:
    def test_computes_exact_price_difference(self):
        a = _product("philips_a", "A", {"price": 79.96})
        b = _product("oralb_b", "B", {"price": 49.99})
        docs = [
            _doc(a.product_id, a.product_name, "price", "stated", 79.96, "$79.96", "e1"),
            _doc(b.product_id, b.product_name, "price", "stated", 49.99, "$49.99", "e2"),
        ]
        result = compare_attribute(a, b, "price", docs)
        assert result.comparable
        assert "A" in result.summary and "B" in result.summary
        assert "$29.97" in result.summary  # exact computed gap, not LLM-guessed
        assert set(result.citation_evidence_ids) == {"e1", "e2"}

    def test_equal_values_phrased_as_tie(self):
        a = _product("philips_a", "A", {"mode_count": 3})
        b = _product("oralb_b", "B", {"mode_count": 3})
        docs = [
            _doc(a.product_id, a.product_name, "mode_count", "stated", 3, "3 modes", "e1"),
            _doc(b.product_id, b.product_name, "mode_count", "stated", 3, "3 modes", "e2"),
        ]
        result = compare_attribute(a, b, "mode_count", docs)
        assert result.comparable
        assert "both have the same stated" in result.summary

    def test_one_side_not_mentioned_is_not_comparable(self):
        a = _product("philips_a", "A", {"price": 79.96})
        b = _product("oralb_b", "B", {})
        docs = [_doc(a.product_id, a.product_name, "price", "stated", 79.96, "$79.96", "e1")]
        result = compare_attribute(a, b, "price", docs)
        assert not result.comparable
        assert "cannot be established" in result.summary

    def test_stated_with_null_value_is_not_comparable(self):
        """stated+value=None (discussed but unnormalizable) must never be
        treated as a numeric zero or skipped silently."""
        a = _product("philips_a", "A", {"price": 79.96})
        b = _product("oralb_b", "B", {"price": None})
        docs = [
            _doc(a.product_id, a.product_name, "price", "stated", 79.96, "$79.96", "e1"),
            _doc(b.product_id, b.product_name, "price", "stated", None, "Contact retailer for pricing.", "e2"),
        ]
        result = compare_attribute(a, b, "price", docs)
        assert not result.comparable


class TestBooleanComparison:
    def test_both_true(self):
        a = _product("philips_a", "A", {"timer": True})
        b = _product("oralb_b", "B", {"timer": True})
        docs = [
            _doc(a.product_id, a.product_name, "timer", "stated", True, "Built-in timer.", "e1"),
            _doc(b.product_id, b.product_name, "timer", "stated", True, "Built-in timer.", "e2"),
        ]
        result = compare_attribute(a, b, "timer", docs)
        assert result.comparable
        assert "Both A and B have timer." == result.summary

    def test_both_false_via_stated_absent(self):
        a = _product("philips_a", "A", {"pressure_sensing": False})
        b = _product("oralb_b", "B", {"pressure_sensing": False})
        docs = [
            _doc(a.product_id, a.product_name, "pressure_sensing", "stated_absent", False, "No sensor.", "e1"),
            _doc(b.product_id, b.product_name, "pressure_sensing", "stated_absent", False, "No sensor.", "e2"),
        ]
        result = compare_attribute(a, b, "pressure_sensing", docs)
        assert result.comparable
        assert "Neither A nor B has pressure sensing." == result.summary

    def test_mixed_true_false(self):
        a = _product("philips_a", "A", {"app_connectivity": True})
        b = _product("oralb_b", "B", {"app_connectivity": False})
        docs = [
            _doc(a.product_id, a.product_name, "app_connectivity", "stated", True, "Pairs with app.", "e1"),
            _doc(b.product_id, b.product_name, "app_connectivity", "stated_absent", False, "No app.", "e2"),
        ]
        result = compare_attribute(a, b, "app_connectivity", docs)
        assert result.comparable
        assert "A has app connectivity, while B does not." == result.summary

    def test_ambiguous_side_is_not_comparable(self):
        a = _product("philips_a", "A", {"timer": True})
        b = _product("oralb_b", "B", {})
        docs = [
            _doc(a.product_id, a.product_name, "timer", "stated", True, "Built-in timer.", "e1"),
            _doc(b.product_id, b.product_name, "timer", "ambiguous", None, "Conflicting timer claims.", "e2"),
        ]
        result = compare_attribute(a, b, "timer", docs)
        assert not result.comparable
