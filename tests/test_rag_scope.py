"""Unit tests for deterministic scope guardrails: out-of-scope source
detection (third-party retailers, non-US pages), the "nothing to scope
the search by" guard that prevents dumping unscoped top-k evidence as
a pseudo-answer, and medical/dental-advice detection (added in response
to a 3C.1 live-evaluation finding)."""
from __future__ import annotations

from src.rag.scope import detect_medical_advice_request, detect_out_of_scope_source, has_no_scoping_signal


class TestDetectOutOfScopeSource:
    def test_amazon_price_request_is_refused(self):
        message = detect_out_of_scope_source("What is the Amazon price of this toothbrush?")
        assert message is not None
        assert "other retailers" in message

    def test_other_known_retailers_are_refused(self):
        for retailer in ("eBay", "Walmart", "Target", "Best Buy", "Costco"):
            assert detect_out_of_scope_source(f"What does {retailer} charge for this?") is not None

    def test_non_us_source_request_is_refused(self):
        message = detect_out_of_scope_source("What does the UK Philips page say about this product?")
        assert message is not None
        assert "other countries" in message

    def test_other_non_us_regions_are_refused(self):
        for phrase in ("the Canada site", "the Australian page", "the India listing"):
            assert detect_out_of_scope_source(f"What does {phrase} say?") is not None

    def test_normal_in_scope_query_is_not_flagged(self):
        assert detect_out_of_scope_source("What is the price of the Sonicare 4100?") is None

    def test_ambiguous_brand_substring_does_not_false_positive(self):
        """"oralb.com" (the approved domain) must not itself trip the
        out-of-scope check."""
        assert detect_out_of_scope_source("Does oralb.com list a timer for the iO Series 9?") is None


class TestHasNoScopingSignal:
    def test_true_when_no_product_and_no_attribute(self):
        assert has_no_scoping_signal("none", [])

    def test_false_when_product_matched_even_without_attribute(self):
        assert not has_no_scoping_signal("matched", [])

    def test_false_when_attribute_detected_even_without_product(self):
        assert not has_no_scoping_signal("none", ["price"])

    def test_true_for_ambiguous_or_not_in_corpus_with_no_attribute(self):
        """These kinds are short-circuited earlier in answer_query before
        this check ever runs, but the predicate itself should still
        treat them as unscoped if ever reached directly."""
        assert has_no_scoping_signal("ambiguous", [])
        assert has_no_scoping_signal("not_in_corpus", [])


class TestDetectMedicalAdviceRequest:
    """Regression coverage for the 3C.1 live-evaluation finding: a
    medical/dental question that names a REAL corpus product has
    product_match.kind=="matched", so it was slipping past the other
    scope checks (which only trigger when NOTHING is scoped) and
    reaching the LLM. These tests target the detector in isolation,
    independent of retrieval/generation."""

    def test_advice_verb_plus_named_condition_is_refused(self):
        assert detect_medical_advice_request("Is the iO Series 10 good for gum disease?") is not None
        assert detect_medical_advice_request("Will this toothbrush treat gingivitis?") is not None
        assert detect_medical_advice_request("Can this toothbrush cure gum disease?") is not None

    def test_named_product_does_not_exempt_a_medical_question(self):
        """The exact scenario the live evaluation found: naming a real
        product must not bypass the medical-advice check."""
        message = detect_medical_advice_request(
            "Is the Oral-B iO Series 4 Clean & Gum Care Rechargeable Electric Toothbrush good for gum disease?"
        )
        assert message is not None

    def test_standalone_medically_recommended_phrase_is_refused(self):
        assert detect_medical_advice_request("Is this toothbrush medically recommended for my gums?") is not None

    def test_other_named_conditions_are_covered(self):
        assert detect_medical_advice_request("Will this heal periodontal disease?") is not None
        assert detect_medical_advice_request("Is this safe for tooth decay?") is not None

    def test_manufacturer_claims_question_is_not_blocked(self):
        """A published-claims question mentioning the same body part
        ("gum health") must remain allowed -- no advice verb is
        present, only a claims-seeking phrase."""
        assert detect_medical_advice_request("What does the manufacturer claim about gum health?") is None

    def test_feature_question_is_not_blocked(self):
        assert detect_medical_advice_request("Does this product have a gum-care mode?") is None

    def test_ordinary_attribute_question_is_not_blocked(self):
        assert detect_medical_advice_request("Does the Sonicare Prestige 9900 have pressure sensing?") is None

    def test_condition_keyword_alone_without_advice_verb_is_not_blocked(self):
        """Merely mentioning a condition name isn't advice-seeking by
        itself -- e.g. a hypothetical claims question about gingivitis
        claims should not be blocked just because the word appears."""
        assert detect_medical_advice_request("What claims does the manufacturer make about gingivitis?") is None

    def test_advice_verb_alone_without_condition_is_not_blocked(self):
        """"good for" alone (no named medical condition) is too weak a
        signal on its own -- e.g. ordinary marketing phrasing."""
        assert detect_medical_advice_request("Is this toothbrush good for travel?") is None
