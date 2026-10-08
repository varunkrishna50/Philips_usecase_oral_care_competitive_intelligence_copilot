"""Tests for the evidence store: deterministic ids, verbatim-substring
enforcement at creation time, and source-text resolution.
"""
from __future__ import annotations

import pytest

from normalization.evidence import (
    STRUCTURED_DATA_SENTINEL,
    EvidenceIntegrityError,
    EvidenceStore,
    resolve_source_text,
)


class TestDeterministicIds:
    def test_id_format(self):
        store = EvidenceStore()
        eid = store.add(
            product_id="philips_hx9990-13", attribute="pressure_sensing", value=True, status="stated",
            raw_value="pressure sensor", evidence_text="pressure sensor",
            source_url="https://example.com/x", source_file="x.html",
            source_section=None, source_section_index=None,
            source_text_for_check="has a pressure sensor built in",
        )
        assert eid == "philips_hx9990-13:pressure_sensing:0"

    def test_ids_increment_per_product_and_attribute(self):
        store = EvidenceStore()
        kwargs = dict(
            source_url="https://example.com/x", source_file="x.html",
            source_section=None, source_section_index=None,
        )
        e0 = store.add(product_id="p1", attribute="claims", value="a", status="stated",
                        raw_value="a", evidence_text="a", source_text_for_check="a b", **kwargs)
        e1 = store.add(product_id="p1", attribute="claims", value="b", status="stated",
                        raw_value="b", evidence_text="b", source_text_for_check="a b", **kwargs)
        e2 = store.add(product_id="p1", attribute="price", value=1, status="stated",
                        raw_value="1", evidence_text="1", source_text_for_check="a b 1", **kwargs)
        assert e0 == "p1:claims:0"
        assert e1 == "p1:claims:1"
        assert e2 == "p1:price:0"

    def test_ids_are_independent_across_products(self):
        store = EvidenceStore()
        kwargs = dict(
            attribute="price", value=1, status="stated", raw_value="1", evidence_text="1",
            source_url="https://example.com/x", source_file="x.html",
            source_section=None, source_section_index=None, source_text_for_check="1",
        )
        e_a = store.add(product_id="product_a", **kwargs)
        e_b = store.add(product_id="product_b", **kwargs)
        assert e_a == "product_a:price:0"
        assert e_b == "product_b:price:0"


class TestVerbatimEnforcementAtCreation:
    def test_add_rejects_evidence_not_in_source(self):
        store = EvidenceStore()
        with pytest.raises(EvidenceIntegrityError):
            store.add(
                product_id="p1", attribute="claims", value="fabricated claim", status="stated",
                raw_value="fabricated claim", evidence_text="fabricated claim",
                source_url="https://example.com/x", source_file="x.html",
                source_section=None, source_section_index=None,
                source_text_for_check="this text does not contain that phrase",
            )

    def test_add_accepts_evidence_that_is_a_true_substring(self):
        store = EvidenceStore()
        eid = store.add(
            product_id="p1", attribute="claims", value="removes 7x more plaque", status="stated",
            raw_value="removes 7x more plaque", evidence_text="removes 7x more plaque",
            source_url="https://example.com/x", source_file="x.html",
            source_section=None, source_section_index=None,
            source_text_for_check="Our brush removes 7x more plaque than a manual toothbrush.",
        )
        assert store.get(eid) is not None

    def test_add_from_section_rejects_evidence_not_in_combined_text(self):
        store = EvidenceStore()
        with pytest.raises(EvidenceIntegrityError):
            store.add_from_section(
                product_id="p1", attribute="timer", value=True, status="stated",
                raw_value="timer", evidence_text="this phrase is nowhere in the section",
                source_url="https://example.com/x", source_file="x.html",
                section_heading="Timer heading", section_index=0,
                section_combined_text="Timer heading\nHas a 2-minute timer.",
            )


class TestResolveSourceText:
    def test_raw_text_source(self):
        record = {"raw_text": "the full page text", "sections": []}
        assert resolve_source_text(record, None, None, tmp_path_dummy()) == "the full page text"

    def test_section_source_concatenates_heading_and_text(self):
        record = {
            "raw_text": "irrelevant",
            "sections": [{"heading": "Pressure Sensor", "text": "Alerts you when brushing too hard."}],
        }
        result = resolve_source_text(record, "Pressure Sensor", 0, tmp_path_dummy())
        assert result == "Pressure Sensor\nAlerts you when brushing too hard."

    def test_section_source_raises_on_heading_mismatch(self):
        record = {
            "raw_text": "irrelevant",
            "sections": [{"heading": "Pressure Sensor", "text": "..."}],
        }
        with pytest.raises(ValueError):
            resolve_source_text(record, "Wrong Heading", 0, tmp_path_dummy())

    def test_section_source_raises_on_out_of_range_index(self):
        record = {"raw_text": "irrelevant", "sections": [{"heading": "A", "text": "..."}]}
        with pytest.raises(ValueError):
            resolve_source_text(record, "A", 5, tmp_path_dummy())

    def test_structured_data_sentinel_concatenates_raw_text_and_html(self, tmp_path):
        html_path = tmp_path / "x.html"
        html_path.write_text('<script>{"price": 19.99}</script>', encoding="utf-8")
        record = {"raw_text": "page text here", "sections": []}
        result = resolve_source_text(record, STRUCTURED_DATA_SENTINEL, None, html_path)
        assert "page text here" in result
        assert '"price": 19.99' in result

    def test_structured_data_sentinel_handles_missing_html_file(self, tmp_path):
        record = {"raw_text": "page text here", "sections": []}
        result = resolve_source_text(record, STRUCTURED_DATA_SENTINEL, None, tmp_path / "missing.html")
        assert result.startswith("page text here")


def tmp_path_dummy():
    from pathlib import Path
    return Path("/nonexistent/does-not-matter-for-this-branch.html")
