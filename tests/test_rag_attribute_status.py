"""Unit tests for status resolution + phrasing templates (3A.12/3B.8)."""
from __future__ import annotations

from src.rag.attribute_status import (
    attribute_label,
    evidence_ids_for,
    resolve_attribute_status,
    status_aware_statement,
)
from src.rag.models import ProductRecord, RetrievalDocument


def _product(attributes=None) -> ProductRecord:
    return ProductRecord(
        product_id="philips_test-product",
        brand="Philips",
        product_name="Test Product",
        product_url="https://www.usa.philips.com/test",
        discovered_url="https://www.usa.philips.com/test",
        source_listing_url="https://www.usa.philips.com/listing",
        retrieved_at="2026-01-01T00:00:00Z",
        attributes=attributes or {},
    )


def _doc(product_id, attribute, status, value, evidence_text, evidence_id="e1") -> RetrievalDocument:
    return RetrievalDocument(
        document_id=evidence_id, evidence_id=evidence_id, product_id=product_id,
        brand="Philips", product_name="Test Product", attribute=attribute, value=value,
        status=status, evidence_text=evidence_text,
        source_url="https://www.usa.philips.com/test", source_section="Features",
        source_file="test.html", retrieval_text=evidence_text,
    )


class TestResolveAttributeStatus:
    def test_no_matching_docs_is_not_mentioned(self):
        product = _product()
        status = resolve_attribute_status(product, "battery_life_days", [])
        assert status.status == "not_mentioned"
        assert status.value is None
        assert status.evidence == []

    def test_stated_with_value_carries_product_value_and_evidence(self):
        product = _product(attributes={"mode_count": 3})
        docs = [_doc(product.product_id, "mode_count", "stated", 3, "3 cleaning modes")]
        status = resolve_attribute_status(product, "mode_count", docs)
        assert status.status == "stated"
        assert status.value == 3
        assert status.evidence == docs

    def test_stated_absent_is_preserved(self):
        product = _product(attributes={"pressure_sensing": False})
        docs = [_doc(product.product_id, "pressure_sensing", "stated_absent", False, "No pressure sensor.")]
        status = resolve_attribute_status(product, "pressure_sensing", docs)
        assert status.status == "stated_absent"

    def test_ignores_docs_for_other_products_or_attributes(self):
        product = _product(attributes={"mode_count": 3})
        docs = [
            _doc("other_product", "mode_count", "stated", 5, "5 modes", evidence_id="e-other"),
            _doc(product.product_id, "price", "stated", 99.0, "$99", evidence_id="e-price"),
        ]
        status = resolve_attribute_status(product, "mode_count", docs)
        assert status.status == "not_mentioned"


class TestStatusAwareStatement:
    def test_not_mentioned_phrasing(self):
        product = _product()
        status = resolve_attribute_status(product, "included_items", [])
        text = status_aware_statement(status)
        assert text == "The available source does not mention included items for Test Product."

    def test_stated_absent_phrasing(self):
        product = _product()
        docs = [_doc(product.product_id, "pressure_sensing", "stated_absent", False, "No pressure sensor.")]
        status = resolve_attribute_status(product, "pressure_sensing", docs)
        text = status_aware_statement(status)
        assert text == "The source explicitly indicates that pressure sensing is not present for Test Product."

    def test_ambiguous_phrasing(self):
        product = _product()
        docs = [_doc(product.product_id, "battery_life_days", "ambiguous", None, "Conflicting battery claims.")]
        status = resolve_attribute_status(product, "battery_life_days", docs)
        text = status_aware_statement(status)
        assert "conflicting or ambiguous" in text
        assert "battery life" in text
        assert "Test Product" in text

    def test_stated_with_null_value_quotes_evidence_verbatim(self):
        product = _product(attributes={"claims": None})
        docs = [_doc(product.product_id, "claims", "stated", None, "Clinically proven results.")]
        status = resolve_attribute_status(product, "claims", docs)
        text = status_aware_statement(status)
        assert '"Clinically proven results."' in text
        assert "does not provide a normalized value" in text

    def test_stated_numeric_value_phrasing(self):
        product = _product(attributes={"mode_count": 3})
        docs = [_doc(product.product_id, "mode_count", "stated", 3, "3 modes")]
        status = resolve_attribute_status(product, "mode_count", docs)
        assert status_aware_statement(status) == "Test Product has 3 for number of brushing modes."

    def test_stated_price_is_formatted_as_currency(self):
        product = _product(attributes={"price": 49.96})
        docs = [_doc(product.product_id, "price", "stated", 49.96, "$49.96")]
        status = resolve_attribute_status(product, "price", docs)
        assert "$49.96" in status_aware_statement(status)

    def test_stated_boolean_true_phrasing(self):
        product = _product(attributes={"timer": True})
        docs = [_doc(product.product_id, "timer", "stated", True, "Built-in timer.")]
        status = resolve_attribute_status(product, "timer", docs)
        assert status_aware_statement(status) == "Test Product has this feature for timer."


class TestEvidenceIdsFor:
    def test_returns_all_evidence_ids_in_order(self):
        product = _product()
        docs = [
            _doc(product.product_id, "included_items", "stated", "Pouch", "1 Pouch", evidence_id="e1"),
            _doc(product.product_id, "included_items", "stated", "Pouch", "1 Pouch", evidence_id="e2"),
        ]
        status = resolve_attribute_status(product, "included_items", docs)
        assert evidence_ids_for(status) == ["e1", "e2"]

    def test_empty_for_not_mentioned(self):
        status = resolve_attribute_status(_product(), "included_items", [])
        assert evidence_ids_for(status) == []


class TestAttributeLabel:
    def test_known_attribute_has_human_label(self):
        assert attribute_label("battery_life_days") == "battery life"

    def test_unknown_attribute_falls_back_to_underscored_name(self):
        assert attribute_label("some_new_field") == "some new field"
