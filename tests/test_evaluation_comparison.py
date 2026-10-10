"""Tests for 3C.9 deterministic comparison evaluation. Gold comparisons
are read from the real, already-loaded product catalog (ProductRecord's
own attributes dict); this file checks both the gold computation itself
and the check against a pipeline-shaped ComparisonResult."""
from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.comparison_eval import check_pipeline_comparison, compute_gold_comparison
from src.rag.models import ProductRecord


def _product(product_id: str, brand: str, **attributes) -> ProductRecord:
    return ProductRecord(
        product_id=product_id, brand=brand, product_name=product_id.upper(),
        product_url="https://example.com", discovered_url="x", source_listing_url="x",
        retrieved_at="2026-01-01T00:00:00Z", attributes=attributes,
    )


def _corpus() -> list[ProductRecord]:
    return [
        _product("p_a", "Philips", price=100.0, pressure_sensing=True),
        _product("p_b", "Oral-B", price=70.0, pressure_sensing=None),
    ]


class TestComputeGoldComparison:
    def test_numeric_both_stated(self):
        corpus = _corpus()
        gold = compute_gold_comparison(corpus, "p_a", "p_b", "price")
        assert gold.comparable
        assert gold.diff == 30.0

    def test_boolean_one_side_not_mentioned_is_not_comparable(self):
        corpus = _corpus()
        gold = compute_gold_comparison(corpus, "p_a", "p_b", "pressure_sensing")
        assert not gold.comparable

    def test_unknown_product_id_is_not_comparable(self):
        corpus = _corpus()
        gold = compute_gold_comparison(corpus, "p_a", "not-a-real-product", "price")
        assert not gold.comparable


@dataclass
class _FakeAttrStatus:
    status: str
    value: object


@dataclass
class _FakeComparisonResult:
    comparable: bool
    status_a: _FakeAttrStatus
    status_b: _FakeAttrStatus


class TestCheckPipelineComparison:
    def test_matching_numeric_diff_passes(self):
        corpus = _corpus()
        gold = compute_gold_comparison(corpus, "p_a", "p_b", "price")
        pipeline = _FakeComparisonResult(True, _FakeAttrStatus("stated", 100.0), _FakeAttrStatus("stated", 70.0))
        result = check_pipeline_comparison(gold, pipeline)
        assert result.passed

    def test_wrong_numeric_diff_fails(self):
        """Guards against an LLM/engine arithmetic error: even if the
        pipeline says "comparable", a wrong value must fail."""
        corpus = _corpus()
        gold = compute_gold_comparison(corpus, "p_a", "p_b", "price")
        pipeline = _FakeComparisonResult(True, _FakeAttrStatus("stated", 999.0), _FakeAttrStatus("stated", 70.0))
        result = check_pipeline_comparison(gold, pipeline)
        assert not result.passed

    def test_comparable_mismatch_fails(self):
        corpus = _corpus()
        gold = compute_gold_comparison(corpus, "p_a", "p_b", "pressure_sensing")  # gold: not comparable
        pipeline = _FakeComparisonResult(True, _FakeAttrStatus("stated", True), _FakeAttrStatus("stated", False))
        result = check_pipeline_comparison(gold, pipeline)
        assert not result.passed

    def test_both_agree_not_comparable_passes(self):
        corpus = _corpus()
        gold = compute_gold_comparison(corpus, "p_a", "p_b", "pressure_sensing")
        pipeline = _FakeComparisonResult(False, _FakeAttrStatus("stated", True), _FakeAttrStatus("not_mentioned", None))
        result = check_pipeline_comparison(gold, pipeline)
        assert result.passed
