"""Tests for product recognition (3A.8/14/15), attribute mapping (3A.9),
and intent classification (3A.13).

Synthetic ProductRecord fixtures exercise the matching algorithm in
isolation; the real-data tests at the bottom dynamically pick products
that actually exist in the corpus rather than hard-coding names that
might not (per the task's explicit instruction).
"""
from __future__ import annotations

from src.rag import corpus, query_understanding as qu
from src.rag.models import ProductRecord


def _product(product_id: str, brand: str, product_name: str) -> ProductRecord:
    return ProductRecord(
        product_id=product_id, brand=brand, product_name=product_name,
        product_url=f"https://example.com/{product_id}", discovered_url="x",
        source_listing_url="x", retrieved_at="2026-01-01T00:00:00Z",
    )


SYNTHETIC_PRODUCTS = [
    _product("oralb_io-series-9-rechargeable-electric-toothbrush", "Oral-B", "iO Series 9 Rechargeable Electric Toothbrush, Aqua Marine"),
    _product("oralb_io-series-9-genius-rechargeable-electric-toothbrush", "Oral-B", "Oral-B iO Series 9 Genius Rechargeable Electric Toothbrush, Sapphire Blue"),
    _product("oralb_io-series-10-rechargeable-electric-toothbrush", "Oral-B", "iO Series 10 Rechargeable Electric Toothbrush, Cosmic Black"),
    _product("philips_hx9990-13", "Philips", "Sonicare Prestige 9900 Rechargeable Toothbrush"),
    _product("philips_hx3641-02", "Philips", "1100 Series Sonic electric toothbrush"),
]


class TestProductRecognitionSynthetic:
    def test_unique_model_number_resolves_to_a_single_match(self):
        m = qu.detect_products("How many modes does iO Series 10 have?", SYNTHETIC_PRODUCTS)
        assert m.kind == "matched"
        assert m.candidates[0].product_id == "oralb_io-series-10-rechargeable-electric-toothbrush"

    def test_exact_requested_product_not_outranked_by_similar_series(self):
        """A query naming Series 10 must never resolve to Series 9 just
        because Series 9 is semantically close (3A.8)."""
        m = qu.detect_products("Tell me about the iO Series 10", SYNTHETIC_PRODUCTS)
        assert m.kind == "matched"
        assert "10" in m.candidates[0].product_id

    def test_family_reference_with_no_number_is_ambiguous(self):
        m = qu.detect_products("iO Series", SYNTHETIC_PRODUCTS)
        assert m.kind == "ambiguous"
        assert len(m.candidates) >= 2
        assert {p.brand for p in m.candidates} == {"Oral-B"}

    def test_series_number_shared_by_a_variant_is_ambiguous(self):
        """"iO Series 9" alone matches both the plain and Genius variant."""
        m = qu.detect_products("iO Series 9", SYNTHETIC_PRODUCTS)
        assert m.kind == "ambiguous"
        ids = {p.product_id for p in m.candidates}
        assert ids == {
            "oralb_io-series-9-rechargeable-electric-toothbrush",
            "oralb_io-series-9-genius-rechargeable-electric-toothbrush",
        }

    def test_generic_category_phrase_spanning_both_brands_is_not_a_product_match(self):
        m = qu.detect_products("Does this electric toothbrush have a timer?", SYNTHETIC_PRODUCTS)
        assert m.kind == "none"

    def test_unrecognized_model_number_is_not_in_corpus(self):
        m = qu.detect_products("Does the Oral-B iO Series 15 have pressure sensing?", SYNTHETIC_PRODUCTS)
        assert m.kind == "not_in_corpus"

    def test_competitor_brand_is_not_in_corpus(self):
        m = qu.detect_products("What about the Colgate Hum?", SYNTHETIC_PRODUCTS)
        assert m.kind == "not_in_corpus"

    def test_no_product_mentioned_at_all(self):
        m = qu.detect_products("How many brushing modes are there?", SYNTHETIC_PRODUCTS)
        assert m.kind == "none"
        assert m.candidates == []

    def test_comparison_resolves_two_distinct_named_products(self):
        matches = qu.detect_comparison_products(
            "Compare Oral-B iO Series 10 and Philips Sonicare Prestige 9900 on brushing modes",
            SYNTHETIC_PRODUCTS,
        )
        ids = {m.candidates[0].product_id for m in matches}
        assert ids == {"oralb_io-series-10-rechargeable-electric-toothbrush", "philips_hx9990-13"}


class TestAttributeMapping:
    def test_pressing_too_hard_maps_to_pressure_sensing(self):
        assert "pressure_sensing" in qu.detect_attributes("What happens if I press too hard?")

    def test_connect_to_phone_maps_to_app_and_bluetooth(self):
        attrs = qu.detect_attributes("Can it connect to my phone?")
        assert "app_connectivity" in attrs
        assert "bluetooth_connectivity" in attrs

    def test_how_long_before_recharge_maps_to_battery_life(self):
        attrs = qu.detect_attributes("How long before I need to recharge?")
        assert "battery_life_days" in attrs
        assert "battery_life_raw" in attrs

    def test_how_many_settings_maps_to_mode_count(self):
        assert "mode_count" in qu.detect_attributes("How many settings does it have?")

    def test_whats_in_the_box_maps_to_included_items(self):
        assert "included_items" in qu.detect_attributes("What comes in the box?")

    def test_no_attribute_detected_for_unrelated_query(self):
        assert qu.detect_attributes("Who founded the company?") == []

    def test_color_query_maps_to_colors(self):
        assert qu.detect_attributes("What color is the Series 10?") == ["colors"]

    def test_only_real_step2_attributes_are_ever_returned(self):
        from src.normalization.models import ATTRIBUTE_FIELDS

        for attrs in qu.ATTRIBUTE_PATTERNS:
            assert attrs in ATTRIBUTE_FIELDS, f"{attrs!r} is not a real Step 2 attribute"


class TestIntentClassification:
    def test_which_products_is_exhaustive(self):
        m = qu.detect_products("Which products have pressure sensing?", SYNTHETIC_PRODUCTS)
        assert qu.classify_intent("Which products have pressure sensing?", m) == "exhaustive"

    def test_compare_keyword_is_comparison(self):
        m = qu.detect_products("Compare iO Series 10 and HX9990/13", SYNTHETIC_PRODUCTS)
        assert qu.classify_intent("Compare iO Series 10 and HX9990/13", m) == "comparison"

    def test_plain_question_is_normal(self):
        m = qu.detect_products("How many modes does iO Series 10 have?", SYNTHETIC_PRODUCTS)
        assert qu.classify_intent("How many modes does iO Series 10 have?", m) == "normal"


# ---------------------------------------------------------------------------
# Real-data tests -- products are picked dynamically from the actual
# corpus, never assumed to exist by name.
# ---------------------------------------------------------------------------

class TestRealDataProductRecognition:
    def test_every_real_product_is_uniquely_resolved_by_its_own_full_name(self):
        products = corpus.load_product_corpus()
        failures = []
        for p in products:
            m = qu.detect_products(p.product_name, products)
            if m.kind != "matched" or m.candidates[0].product_id != p.product_id:
                failures.append((p.product_id, m.kind, [c.product_id for c in m.candidates]))
        assert not failures, f"products not uniquely resolved by their own name: {failures}"

    def test_ambiguous_family_query_only_if_corpus_actually_has_a_shared_family(self):
        products = corpus.load_product_corpus()
        m = qu.detect_products("iO Series", products)
        io_series_products = [p for p in products if "io-series" in p.product_id]
        if len(io_series_products) >= 2:
            assert m.kind == "ambiguous"
        # if Step 1/2 output ever changes shape, this test degrades
        # gracefully rather than asserting on a family that may not exist.

    def test_real_exhaustive_query_attribute_is_a_real_column(self):
        from src.normalization.models import ATTRIBUTE_FIELDS

        m = qu.detect_products("Which products have a timer?", corpus.load_product_corpus())
        attrs = qu.detect_attributes("Which products have a timer?")
        assert all(a in ATTRIBUTE_FIELDS for a in attrs)


class TestDetectExhaustiveConditionValue:
    """Regression tests for the 3C.1-found bug: the exhaustive path
    used to hardcode condition_value=True for every attribute, so a
    numeric exhaustive query ("7 brushing modes") silently compared
    mode_count to True instead of 7, returning zero matches."""

    def test_boolean_attribute_still_defaults_to_true(self):
        assert qu.detect_exhaustive_condition_value("Which products have pressure sensing?", "pressure_sensing") is True
        assert qu.detect_exhaustive_condition_value("Which products support app connectivity?", "app_connectivity") is True

    def test_numeric_attribute_extracts_the_stated_number(self):
        value = qu.detect_exhaustive_condition_value("Which products have 7 brushing modes?", "mode_count")
        assert value == 7
        assert isinstance(value, int)

    def test_numeric_attribute_with_no_number_returns_none(self):
        """A numeric attribute named with no actual number in the query
        has no well-defined condition -- must not fall back to True
        (which would silently compare mode_count to a boolean)."""
        assert qu.detect_exhaustive_condition_value("Which products have brushing modes?", "mode_count") is None

    def test_numeric_attribute_handles_decimal_values(self):
        value = qu.detect_exhaustive_condition_value("Which products cost 49.99?", "price")
        assert value == 49.99

    def test_non_exhaustive_numeric_attributes_are_unaffected(self):
        """Only the small, explicit NUMERIC_EXHAUSTIVE_ATTRIBUTES set
        changes behavior -- an attribute not in that set still defaults
        to True exactly as before."""
        assert qu.detect_exhaustive_condition_value("Which products have a timer?", "timer") is True
