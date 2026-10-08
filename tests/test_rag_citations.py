"""Unit tests for structured segments, citation validation (3B.5/3B.6),
and source rendering (3B.7)."""
from __future__ import annotations

from src.rag.citations import (
    Segment,
    assign_markers,
    render_answer_text,
    render_sources_section,
    validate_segment,
    validate_segments,
)
from src.rag.models import ProductRecord, RetrievalDocument


def _product(product_id, name) -> ProductRecord:
    return ProductRecord(
        product_id=product_id, brand="Philips", product_name=name,
        product_url="https://www.usa.philips.com/test", discovered_url="https://www.usa.philips.com/test",
        source_listing_url="https://www.usa.philips.com/listing", retrieved_at="2026-01-01T00:00:00Z",
    )


def _doc(product_id, product_name, attribute, text, evidence_id, value=None, status="stated") -> RetrievalDocument:
    return RetrievalDocument(
        document_id=evidence_id, evidence_id=evidence_id, product_id=product_id,
        brand="Philips", product_name=product_name, attribute=attribute, value=value,
        status=status, evidence_text=text, source_url=f"https://www.usa.philips.com/{product_id}",
        source_section="Features", source_file="t.html", retrieval_text=text,
    )


PRODUCT_A = _product("philips_prestige-9900", "Sonicare Prestige 9900")
PRODUCT_B = _product("philips_series-4100", "Sonicare 4100")
CORPUS = [PRODUCT_A, PRODUCT_B]


class TestValidatedSegmentCarriesTrustedFlag:
    """3C.1 needs to tell trusted (always-valid-by-construction) segments
    apart from LLM-authored ones AFTER validation, to avoid reporting an
    inflated offline citation-validity rate."""

    def test_trusted_segment_result_is_marked_trusted(self):
        seg = Segment(text="x", citation_ids=[], trusted=True)
        result = validate_segment(seg, {}, CORPUS)
        assert result.trusted is True

    def test_untrusted_segment_result_is_marked_not_trusted(self):
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        seg = Segment(text="Sonicare Prestige 9900 has a price of $79.96.", citation_ids=["e1"])
        result = validate_segment(seg, {"e1": ev}, CORPUS)
        assert result.trusted is False

    def test_rejected_untrusted_segment_is_still_marked_not_trusted(self):
        seg = Segment(text="Unsupported.", citation_ids=["missing"])
        result = validate_segment(seg, {}, CORPUS)
        assert result.trusted is False


class TestTrustedSegmentsSkipValidation:
    def test_trusted_segment_with_no_citations_is_still_valid(self):
        """A not_mentioned disclosure has nothing to cite by definition."""
        seg = Segment(text="The available source does not mention X.", citation_ids=[], trusted=True)
        result = validate_segment(seg, {}, CORPUS)
        assert result.valid
        assert result.citation_ids == []

    def test_trusted_segment_with_computed_number_not_in_any_evidence_is_still_valid(self):
        """Comparison summaries state a computed difference that isn't a
        verbatim quote of any single evidence_text -- correct by
        construction, must not be rejected by the numeric check."""
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        seg = Segment(text="A has a price of $79.96 while B has $49.99 -- a difference of $29.97.",
                      citation_ids=["e1"], trusted=True)
        result = validate_segment(seg, {"e1": ev}, CORPUS)
        assert result.valid


class TestUntrustedSegmentValidation:
    def test_valid_when_citation_exists_and_matches_product_and_attribute(self):
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        seg = Segment(text="Sonicare Prestige 9900 has a price of $79.96.", citation_ids=["e1"])
        result = validate_segment(seg, {"e1": ev}, CORPUS, allowed_product_ids={PRODUCT_A.product_id}, allowed_attributes=["price"])
        assert result.valid
        assert result.citation_ids == ["e1"]

    def test_rejects_unknown_citation_id(self):
        seg = Segment(text="Some claim.", citation_ids=["does-not-exist"])
        result = validate_segment(seg, {}, CORPUS)
        assert not result.valid
        assert "unknown citation id" in result.reasons[0]

    def test_rejects_segment_with_no_citations(self):
        seg = Segment(text="An unsupported claim with no source.", citation_ids=[])
        result = validate_segment(seg, {}, CORPUS)
        assert not result.valid

    def test_rejects_citation_belonging_to_wrong_product(self):
        ev = _doc(PRODUCT_B.product_id, PRODUCT_B.product_name, "price", "$49.99", "e1", value=49.99)
        seg = Segment(text="Sonicare 4100 costs $49.99.", citation_ids=["e1"])
        result = validate_segment(seg, {"e1": ev}, CORPUS, allowed_product_ids={PRODUCT_A.product_id})
        assert not result.valid
        assert any("not an expected product" in r for r in result.reasons)

    def test_rejects_citation_with_wrong_attribute(self):
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "mode_count", "3 modes", "e1", value=3)
        seg = Segment(text="Sonicare Prestige 9900 has 3 modes.", citation_ids=["e1"])
        result = validate_segment(seg, {"e1": ev}, CORPUS, allowed_attributes=["price"])
        assert not result.valid
        assert any("not in requested" in r for r in result.reasons)

    def test_rejects_when_text_names_a_different_product_than_cited(self):
        """The LLM claiming a fact about Product B while only citing
        Product A's evidence must be caught -- this is the core
        hallucination guard."""
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        seg = Segment(text="Sonicare 4100 has a price of $79.96.", citation_ids=["e1"])
        result = validate_segment(seg, {"e1": ev}, CORPUS)
        assert not result.valid
        assert any("names product" in r for r in result.reasons)

    def test_rejects_numeric_claim_not_backed_by_any_cited_evidence(self):
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        seg = Segment(text="Sonicare Prestige 9900 has a price of $999.99.", citation_ids=["e1"])
        result = validate_segment(seg, {"e1": ev}, CORPUS)
        assert not result.valid
        assert any("999.99" in r for r in result.reasons)

    def test_digits_in_a_numeric_product_name_are_not_treated_as_unsupported_claims(self):
        """Regression: "Sonicare 4100" legitimately contains a digit that
        is part of the product's NAME, not a separate numeric fact --
        restating it must not be flagged as an unsupported numeric claim
        just because the cited evidence_text (a price string) doesn't
        separately contain "4100"."""
        ev = _doc(PRODUCT_B.product_id, PRODUCT_B.product_name, "price", "$49.99", "e1", value=49.99)
        seg = Segment(text="Sonicare 4100 has a price of $49.99.", citation_ids=["e1"])
        result = validate_segment(seg, {"e1": ev}, CORPUS)
        assert result.valid

    def test_abbreviated_product_name_still_matches_by_digit_not_full_string(self):
        """Regression found via a real LLM call: the product's full name
        is "Sonicare Prestige 9900 Rechargeable Toothbrush", but the LLM
        naturally shortened it to "Sonicare Prestige 9900" -- a literal
        full-name substring match would miss this and wrongly flag
        "9900" as an unsupported claim. Matching by digit-token (not
        full-string containment) must tolerate the paraphrase."""
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$379.96", "e1", value=379.96)
        seg = Segment(text="The price of the Sonicare Prestige 9900 is $379.96 USD.", citation_ids=["e1"])
        result = validate_segment(seg, {"e1": ev}, CORPUS)
        assert result.valid

    def test_partial_hallucinated_handle_among_valid_ones_keeps_only_resolved(self):
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        seg = Segment(text="Sonicare Prestige 9900 has a price of $79.96.", citation_ids=["e1", "e-fake"])
        result = validate_segment(seg, {"e1": ev}, CORPUS)
        assert not result.valid  # unknown id present -> whole segment rejected, never silently trimmed


class TestValidateSegments:
    def test_preserves_order_and_mixes_valid_invalid(self):
        ev = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        segs = [
            Segment(text="Sonicare Prestige 9900 has a price of $79.96.", citation_ids=["e1"]),
            Segment(text="Unsupported claim.", citation_ids=["missing"]),
        ]
        results = validate_segments(segs, {"e1": ev}, CORPUS)
        assert [r.valid for r in results] == [True, False]


class TestAssignMarkersAndRendering:
    def test_markers_assigned_in_first_appearance_order_valid_only(self):
        ev1 = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        ev2 = _doc(PRODUCT_B.product_id, PRODUCT_B.product_name, "price", "$49.99", "e2", value=49.99)
        valid1 = validate_segment(Segment("A", ["e1"], trusted=True), {"e1": ev1}, CORPUS)
        invalid = validate_segment(Segment("B", ["missing"]), {}, CORPUS)
        valid2 = validate_segment(Segment("C", ["e2"], trusted=True), {"e2": ev2}, CORPUS)
        markers = assign_markers([valid1, invalid, valid2])
        assert markers == {"e1": "S1", "e2": "S2"}

    def test_render_answer_text_skips_invalid_segments(self):
        ev1 = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        valid = validate_segment(Segment("A costs $79.96.", ["e1"], trusted=True), {"e1": ev1}, CORPUS)
        invalid = validate_segment(Segment("Bad claim.", ["missing"]), {}, CORPUS)
        markers = assign_markers([valid, invalid])
        text = render_answer_text([valid, invalid], markers)
        assert text == "A costs $79.96. [S1]"

    def test_render_sources_section_lists_product_and_url_never_llm_generated(self):
        ev1 = _doc(PRODUCT_A.product_id, PRODUCT_A.product_name, "price", "$79.96", "e1", value=79.96)
        markers = {"e1": "S1"}
        text = render_sources_section(markers, {"e1": ev1})
        assert text.startswith("Sources\n\n")
        assert "[S1] Sonicare Prestige 9900 -- official product page" in text
        assert ev1.source_url in text

    def test_render_sources_section_empty_when_no_markers(self):
        assert render_sources_section({}, {}) == ""
