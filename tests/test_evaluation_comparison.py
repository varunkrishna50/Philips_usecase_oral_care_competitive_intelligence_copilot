"""Tests for 3C.9 deterministic comparison evaluation. Gold comparisons
are computed independently of src.rag.comparison_engine; this file
checks both the gold computation itself and the check against a
pipeline-shaped ComparisonResult."""
from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.comparison_eval import check_pipeline_comparison, compute_gold_comparison
from src.evaluation.gold_labels import GoldCorpus, GoldEvidence, GoldProduct


def _corpus() -> GoldCorpus:
    products = [
        GoldProduct(product_id="p_a", brand="Philips", product_name="A", product_url="https://example.com",
                    attributes={"price": 100.0, "pressure_sensing": True}),
        GoldProduct(product_id="p_b", brand="Oral-B", product_name="B", product_url="https://example.com",
                    attributes={"price": 70.0, "pressure_sensing": None}),
    ]
    evidence = [
        GoldEvidence(evidence_id="p_a:price:0", product_id="p_a", attribute="price", value=100.0,
                     status="stated", evidence_text="$100.00", source_url="https://example.com"),
        GoldEvidence(evidence_id="p_b:price:0", product_id="p_b", attribute="price", value=70.0,
                     status="stated", evidence_text="$70.00", source_url="https://example.com"),
        GoldEvidence(evidence_id="p_a:pressure_sensing:0", product_id="p_a", attribute="pressure_sensing", value=True,
                     status="stated", evidence_text="Has a pressure sensor.", source_url="https://example.com"),
    ]
    return GoldCorpus(products=products, evidence=evidence)


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
