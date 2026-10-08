"""Tests for 3C.4/3C.9 exhaustive-completeness calculation, independent
of RetrievalEngine's own exhaustive_query (the expected set comes from
gold_labels.py; only the ACTUAL side under test uses pipeline output)."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.evaluation.exhaustive_eval import check_exhaustive_completeness, gold_exhaustive_expected
from src.evaluation.gold_labels import GoldCorpus, GoldProduct


def _corpus() -> GoldCorpus:
    products = [
        GoldProduct(product_id=f"p{i}", brand="Oral-B", product_name=f"Product {i}",
                    product_url="https://example.com", attributes={"pressure_sensing": i < 3, "mode_count": 7 if i == 4 else 5})
        for i in range(5)
    ]
    return GoldCorpus(products=products, evidence=[])


@dataclass
class _FakeMatch:
    product: object
    evidence: list = field(default_factory=lambda: [object()])


class TestGoldExhaustiveExpected:
    def test_boolean_condition(self):
        corpus = _corpus()
        expected = gold_exhaustive_expected(corpus, "pressure_sensing", True)
        assert expected == {"p0", "p1", "p2"}

    def test_numeric_condition(self):
        corpus = _corpus()
        expected = gold_exhaustive_expected(corpus, "mode_count", 7)
        assert expected == {"p4"}


class TestCheckExhaustiveCompleteness:
    def test_complete_match(self):
        corpus = _corpus()
        expected = gold_exhaustive_expected(corpus, "pressure_sensing", True)
        matches = [_FakeMatch(product=p) for p in corpus.products if p.product_id in expected]
        result = check_exhaustive_completeness(expected, matches)
        assert result.complete
        assert result.recall == 1.0
        assert result.missing_product_ids == []

    def test_missing_product_is_detected(self):
        corpus = _corpus()
        expected = gold_exhaustive_expected(corpus, "pressure_sensing", True)
        matches = [_FakeMatch(product=corpus.products[0])]  # only 1 of 3
        result = check_exhaustive_completeness(expected, matches)
        assert not result.complete
        assert result.recall < 1.0
        assert set(result.missing_product_ids) == {"p1", "p2"}

    def test_extra_product_is_detected(self):
        corpus = _corpus()
        expected = gold_exhaustive_expected(corpus, "pressure_sensing", True)
        matches = [_FakeMatch(product=p) for p in corpus.products]  # all 5, but only 3 expected
        result = check_exhaustive_completeness(expected, matches)
        assert not result.complete
        assert set(result.extra_product_ids) == {"p3", "p4"}

    def test_match_without_evidence_is_not_complete(self):
        corpus = _corpus()
        expected = {"p0"}
        matches = [_FakeMatch(product=corpus.products[0], evidence=[])]
        result = check_exhaustive_completeness(expected, matches)
        assert not result.every_match_has_evidence
        assert not result.complete

    def test_empty_expected_set_is_vacuously_complete(self):
        result = check_exhaustive_completeness(set(), [])
        assert result.complete
        assert result.recall == 1.0

    def test_numeric_condition_real_bug_scenario(self):
        """Mirrors the real exhaustive_modes7_014 case: the expected set
        (mode_count==7) is non-empty, but if the pipeline wrongly scans
        for mode_count==True (the actual bug), it returns zero matches --
        this must be reported as incomplete, not silently passed."""
        corpus = _corpus()
        expected = gold_exhaustive_expected(corpus, "mode_count", 7)
        result = check_exhaustive_completeness(expected, [])  # simulates the buggy zero-match output
        assert not result.complete
        assert result.recall == 0.0
