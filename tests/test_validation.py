"""Tests for the independent, mechanical validation pass: evidence
integrity, verbatim re-verification, and cross-product detection.

These construct NormalizedProduct/EvidenceRef objects directly (rather
than going through the full extraction pipeline) so each failure mode
can be triggered in isolation.
"""
from __future__ import annotations

from normalization.evidence import EvidenceStore
from normalization.models import AttributeValue, EvidenceRef, NormalizedProduct, not_mentioned
from normalization.validation import validate

from normalization_fixtures import make_raw_record


def _minimal_product(**overrides) -> NormalizedProduct:
    fields = dict(
        product_id="philips_test-1",
        brand="Philips",
        product_name="Test Toothbrush",
        product_url="https://example.com/philips/test-1",
        discovered_url="https://example.com/philips/test-1",
        source_listing_url="https://example.com/philips/listing",
        retrieved_at="2026-01-01T00:00:00+00:00",
        raw_html_path="irrelevant.html",
        content_hash="0" * 64,
    )
    fields.update(overrides)
    return NormalizedProduct(**fields)


def _well_formed_store_and_raw_index(tmp_path):
    record = make_raw_record(
        tmp_path,
        brand="Philips",
        slug="test-1",
        sections=[{"heading": "Pressure Sensor", "text": "Alerts you when brushing too hard."}],
    )
    store = EvidenceStore()
    eid = store.add_from_section(
        product_id="philips_test-1", attribute="pressure_sensing", value=True, status="stated",
        raw_value="brushing too hard", evidence_text="Alerts you when brushing too hard.",
        source_url=record["product_url"], source_file=record["raw_html_path"],
        section_heading="Pressure Sensor", section_index=0,
        section_combined_text="Pressure Sensor\nAlerts you when brushing too hard.",
    )
    product = _minimal_product(
        pressure_sensing=AttributeValue(value=True, status="stated", evidence_ids=[eid]),
    )
    raw_index = {"philips_test-1": record}
    return product, store, raw_index


class TestWellFormedPasses:
    def test_valid_product_and_evidence_pass(self, tmp_path):
        product, store, raw_index = _well_formed_store_and_raw_index(tmp_path)
        errors = validate([product], store, raw_index)
        assert errors == []


class TestIdentityValidation:
    def test_duplicate_product_id_detected(self, tmp_path):
        p1 = _minimal_product()
        p2 = _minimal_product(product_name="Different Name")  # same product_id
        store = EvidenceStore()
        errors = validate([p1, p2], store, {})
        assert any("duplicate product_id" in e for e in errors)

    def test_missing_brand_detected(self):
        p = _minimal_product(brand="")
        errors = validate([p], EvidenceStore(), {})
        assert any("missing brand" in e for e in errors)

    def test_missing_product_url_detected(self):
        p = _minimal_product(product_url="")
        errors = validate([p], EvidenceStore(), {})
        assert any("missing product_url" in e for e in errors)


class TestEvidenceIntegrityValidation:
    def test_dangling_evidence_id_detected(self):
        p = _minimal_product(
            pressure_sensing=AttributeValue(value=True, status="stated", evidence_ids=["nonexistent:pressure_sensing:0"]),
        )
        errors = validate([p], EvidenceStore(), {})
        assert any("does not exist in evidence.json" in e for e in errors)

    def test_not_mentioned_with_evidence_ids_is_flagged(self):
        store = EvidenceStore()
        p = _minimal_product(
            pressure_sensing=AttributeValue(value=None, status="not_mentioned", evidence_ids=["philips_test-1:pressure_sensing:0"]),
        )
        errors = validate([p], store, {})
        assert any("not_mentioned but has evidence_ids" in e for e in errors)

    def test_stated_without_evidence_ids_is_flagged(self):
        p = _minimal_product(
            pressure_sensing=AttributeValue(value=True, status="stated", evidence_ids=[]),
        )
        errors = validate([p], EvidenceStore(), {})
        assert any("has no evidence_ids" in e for e in errors)


class TestVerbatimReValidation:
    def test_evidence_text_not_in_claimed_source_is_caught(self, tmp_path):
        """Simulates a bug where evidence_text drifted from its claimed
        source after creation -- the independent re-check (which re-reads
        from the raw record, not from the in-memory object that created
        the evidence) must catch this even though EvidenceStore.add()
        itself would have rejected it at creation time.
        """
        record = make_raw_record(
            tmp_path, brand="Philips", slug="test-1",
            sections=[{"heading": "Pressure Sensor", "text": "Alerts you when brushing too hard."}],
        )
        store = EvidenceStore()
        bad_ref = EvidenceRef(
            id="philips_test-1:pressure_sensing:0", product_id="philips_test-1", attribute="pressure_sensing",
            value=True, status="stated", raw_value="fabricated",
            evidence_text="this sentence was never on the page",
            source_url=record["product_url"], source_file=record["raw_html_path"],
            source_section="Pressure Sensor", source_section_index=0,
        )
        store._items[bad_ref.id] = bad_ref  # bypass add()'s own check, as if a bug had let this through
        p = _minimal_product(
            pressure_sensing=AttributeValue(value=True, status="stated", evidence_ids=[bad_ref.id]),
        )
        errors = validate([p], store, {"philips_test-1": record})
        assert any("NOT a verbatim substring" in e for e in errors)

    def test_section_index_heading_mismatch_is_caught(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="test-1",
            sections=[{"heading": "Pressure Sensor", "text": "Alerts you when brushing too hard."}],
        )
        store = EvidenceStore()
        bad_ref = EvidenceRef(
            id="philips_test-1:pressure_sensing:0", product_id="philips_test-1", attribute="pressure_sensing",
            value=True, status="stated", raw_value="x", evidence_text="x",
            source_url=record["product_url"], source_file=record["raw_html_path"],
            source_section="A Different Heading Entirely", source_section_index=0,
        )
        store._items[bad_ref.id] = bad_ref
        p = _minimal_product(pressure_sensing=AttributeValue(value=True, status="stated", evidence_ids=[bad_ref.id]))
        errors = validate([p], store, {"philips_test-1": record})
        # resolve_source_text() itself raises ValueError on a heading
        # mismatch; the validator surfaces that as a resolution failure.
        assert any("could not resolve claimed source" in e for e in errors)


class TestNoCrossProductEvidence:
    def test_evidence_pointing_to_a_different_products_url_is_caught(self, tmp_path):
        record = make_raw_record(tmp_path, brand="Philips", slug="test-1")
        store = EvidenceStore()
        bad_ref = EvidenceRef(
            id="philips_test-1:price:0", product_id="philips_test-1", attribute="price",
            value=29.96, status="stated", raw_value="29.96", evidence_text="29.96",
            source_url="https://example.com/philips/SOME-OTHER-PRODUCT",  # wrong product's URL
            source_file=record["raw_html_path"], source_section=None, source_section_index=None,
        )
        store._items[bad_ref.id] = bad_ref
        p = _minimal_product(price=AttributeValue(value=29.96, status="stated", evidence_ids=[bad_ref.id]))
        errors = validate([p], store, {"philips_test-1": record})
        assert any("possible cross-product evidence" in e for e in errors)
