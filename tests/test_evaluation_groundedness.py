"""Tests for the independent groundedness checker (3C.7). Must not
import src.rag.citations, and must flag issues in ALL segments
including ones that would be marked `trusted` by the real pipeline --
this checker has no concept of `trusted` at all, by design, which is
exactly how it independently verifies a claim a trusted-skip could let
slip through."""
from __future__ import annotations

import ast
import inspect

from src.evaluation import groundedness
from src.evaluation.comparison_eval import GoldComparison
from src.evaluation.gold_labels import GoldCorpus, GoldEvidence, GoldProduct


class TestIndependenceFromCitationsModule:
    def test_module_does_not_import_citations(self):
        """Checks actual import statements (via AST), not the
        docstring's prose explanation of this independence requirement
        -- which itself legitimately mentions "src.rag.citations"."""
        tree = ast.parse(inspect.getsource(groundedness))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        assert not any("citations" in m for m in imported_modules)


def _corpus() -> GoldCorpus:
    products = [
        GoldProduct(product_id="p_a", brand="Philips", product_name="Sonicare Prestige 9900",
                    product_url="https://example.com/a", attributes={"price": 379.96, "mode_count": 5}),
        GoldProduct(product_id="p_b", brand="Oral-B", product_name="iO Series 10",
                    product_url="https://example.com/b", attributes={"price": 399.99, "mode_count": 7}),
    ]
    evidence = [
        GoldEvidence(evidence_id="p_a:price:0", product_id="p_a", attribute="price", value=379.96,
                     status="stated", evidence_text="$379.96", source_url="https://example.com/a"),
        GoldEvidence(evidence_id="p_b:price:0", product_id="p_b", attribute="price", value=399.99,
                     status="stated", evidence_text="$399.99", source_url="https://example.com/b"),
    ]
    return GoldCorpus(products=products, evidence=evidence)


class TestCheckSegment:
    def test_grounded_claim_has_no_issues(self):
        corpus = _corpus()
        result = groundedness.check_segment("Sonicare Prestige 9900 has a price of $379.96.", corpus)
        assert result.grounded

    def test_flags_a_deliberately_fabricated_price(self):
        """The core requirement: a wrong numeric claim about a real
        product must be caught, with no dependence on whether the
        generation pipeline would have marked this segment trusted."""
        corpus = _corpus()
        result = groundedness.check_segment("Sonicare Prestige 9900 has a price of $999.99.", corpus)
        assert not result.grounded
        assert any(i.kind == "unsupported_price" for i in result.issues)

    def test_flags_unsupported_product_name_outside_allowed_set(self):
        corpus = _corpus()
        result = groundedness.check_segment("iO Series 10 has a price of $399.99.", corpus, allowed_product_ids={"p_a"})
        assert any(i.kind == "unsupported_product_name" for i in result.issues)

    def test_does_not_flag_digits_in_product_name_itself(self):
        corpus = _corpus()
        result = groundedness.check_segment("iO Series 10 has a price of $399.99.", corpus)
        assert result.grounded

    def test_multi_product_segment_checks_against_the_union_not_each_individually(self):
        """A segment naming two products must not be flagged just
        because product A's number doesn't appear in product B's own
        evidence -- regression for a real bug found during 3C.1
        development."""
        corpus = _corpus()
        text = "iO Series 10 has a price of $399.99, while Sonicare Prestige 9900 has $379.96."
        result = groundedness.check_segment(text, corpus)
        assert result.grounded

    def test_difference_number_is_excluded_from_the_generic_numeric_check(self):
        corpus = _corpus()
        text = "iO Series 10 has $399.99, while Sonicare Prestige 9900 has $379.96 -- a difference of $20.03."
        result = groundedness.check_segment(text, corpus)
        assert result.grounded


class TestCheckBooleanClaim:
    def test_matching_true_claim_is_not_flagged(self):
        corpus = _corpus()
        product = corpus.products[0]
        issue = groundedness.check_boolean_claim("This product has pressure sensing.", product, "pressure_sensing", True)
        assert issue is None

    def test_mismatched_claim_is_flagged(self):
        corpus = _corpus()
        product = corpus.products[0]
        issue = groundedness.check_boolean_claim("This product has pressure sensing.", product, "pressure_sensing", False)
        assert issue is not None
        assert issue.kind == "unsupported_boolean_attribute"

    def test_negated_claim_matches_false_gold(self):
        corpus = _corpus()
        product = corpus.products[0]
        issue = groundedness.check_boolean_claim("This product does not have pressure sensing.", product, "pressure_sensing", False)
        assert issue is None

    def test_no_ground_truth_returns_none(self):
        corpus = _corpus()
        product = corpus.products[0]
        issue = groundedness.check_boolean_claim("This product has pressure sensing.", product, "pressure_sensing", None)
        assert issue is None


class TestCheckComparisonClaims:
    def test_correct_difference_not_flagged(self):
        corpus = _corpus()
        gold = GoldComparison(attribute="price", status_a="stated", status_b="stated", value_a=379.96, value_b=399.99, comparable=True, diff=20.03)
        issues = groundedness.check_comparison_claims(
            "iO Series 10 has $399.99 while Sonicare Prestige 9900 has $379.96 -- a difference of $20.03.",
            gold, corpus.products[1], corpus.products[0],
        )
        assert issues == []

    def test_wrong_difference_is_flagged(self):
        corpus = _corpus()
        gold = GoldComparison(attribute="price", status_a="stated", status_b="stated", value_a=379.96, value_b=399.99, comparable=True, diff=20.03)
        issues = groundedness.check_comparison_claims(
            "iO Series 10 has $399.99 while Sonicare Prestige 9900 has $379.96 -- a difference of $5.00.",
            gold, corpus.products[1], corpus.products[0],
        )
        assert len(issues) == 1
        assert issues[0].kind == "unsupported_comparison"
