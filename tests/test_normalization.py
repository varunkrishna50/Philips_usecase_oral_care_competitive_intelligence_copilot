"""Unit tests for Step 2 attribute extraction, the four-status model,
cross-product/cross-sell exclusion, product IDs, and CSV serialization.

All tests use synthetic fixtures built by normalization_fixtures.py --
none of them depend on live website content. A final integration test
runs the real Step 1 output through the full pipeline + validator as a
regression check.
"""
from __future__ import annotations

from pathlib import Path

from normalization import oralb, philips, utils
from normalization.evidence import EvidenceStore
from normalization.filters import usable_sections
from normalization.normalize import index_raw_records, load_raw_records, normalize_all
from normalization.validation import validate

from normalization_fixtures import make_raw_record, section


# ---------------------------------------------------------------------------
# Price extraction
# ---------------------------------------------------------------------------

class TestPhilipsPriceExtraction:
    def test_price_from_json_ld(self, tmp_path):
        record = make_raw_record(tmp_path, brand="Philips", slug="p1", json_ld_price=379.96, json_ld_currency="USD")
        store, warnings = EvidenceStore(), []
        price, currency, price_type = utils.extract_price(record, "philips_p1", store, warnings)
        assert price.value == 379.96
        assert price.status == "stated"
        assert currency.value == "USD"
        assert price_type.value == "listed"
        assert warnings == []

    def test_price_not_mentioned_when_no_signal(self, tmp_path):
        record = make_raw_record(tmp_path, brand="Philips", slug="p1")
        store, warnings = EvidenceStore(), []
        price, currency, price_type = utils.extract_price(record, "philips_p1", store, warnings)
        assert price.status == "not_mentioned"
        assert price.value is None
        assert any(w.code == "no_price_signal" for w in warnings)


class TestOralBPriceExtraction:
    def test_string_typed_json_ld_price_and_price_text_agree(self, tmp_path):
        record = make_raw_record(tmp_path, brand="Oral-B", slug="o1", json_ld_price="169.99", price_text="$169.99")
        store, warnings = EvidenceStore(), []
        price, currency, price_type = utils.extract_price(record, "oralb_o1", store, warnings)
        assert price.value == 169.99
        assert price.status == "stated"
        assert currency.value == "USD"

    def test_bundle_price_elsewhere_on_page_is_never_used_as_standalone_price(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1", json_ld_price="169.99", price_text="$169.99",
            raw_text="Purchase Type: Toothbrush Only $169.99 Toothbrush + Brush Heads $329.99 Save $160.00",
        )
        store, warnings = EvidenceStore(), []
        price, _, _ = utils.extract_price(record, "oralb_o1", store, warnings)
        assert price.value == 169.99
        assert not any(w.code == "price_conflict" for w in warnings)

    def test_out_of_stock_price_is_flagged_unavailable_not_silently_listed(self, tmp_path):
        """A product can carry a real catalog price in its JSON-LD offer
        while schema.org/OutOfStock says it can't currently be bought --
        price_type must say "unavailable", not silently "listed", even
        though the price value itself is still preserved as useful
        (last-known) information.
        """
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1", json_ld_price="399.99",
            json_ld_availability="https://schema.org/OutOfStock",
        )
        store, warnings = EvidenceStore(), []
        price, currency, price_type = utils.extract_price(record, "oralb_o1", store, warnings)
        assert price.value == 399.99  # still preserved -- it's real, just not current
        assert price.status == "stated"
        assert price_type.value == "unavailable"
        assert any(w.code == "price_unavailable" for w in warnings)

    def test_in_stock_price_is_listed(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1", json_ld_price="169.99",
            json_ld_availability="https://schema.org/InStock",
        )
        store, warnings = EvidenceStore(), []
        _price, _currency, price_type = utils.extract_price(record, "oralb_o1", store, warnings)
        assert price_type.value == "listed"
        assert not any(w.code == "price_unavailable" for w in warnings)

    def test_genuine_price_conflict_is_ambiguous_with_both_values_preserved(self, tmp_path):
        record = make_raw_record(tmp_path, brand="Oral-B", slug="o1", json_ld_price="169.99", price_text="$179.99")
        store, warnings = EvidenceStore(), []
        price, currency, price_type = utils.extract_price(record, "oralb_o1", store, warnings)
        assert price.status == "ambiguous"
        assert price.value is None
        assert len(price.evidence_ids) == 2
        assert {store.get(e).value for e in price.evidence_ids} == {169.99, 179.99}
        assert any(w.code == "price_conflict" for w in warnings)
        assert price_type.status == "ambiguous"


# ---------------------------------------------------------------------------
# Status model: stated / not_mentioned / stated_absent / ambiguous /
# "stated but unnormalizable"
# ---------------------------------------------------------------------------

class TestStatusModel:
    def test_stated_absent_for_explicit_negation(self, tmp_path):
        # The heading itself restates "Timer" (as a real FAQ heading like
        # "Does the Series 2 have a pressure sensor?" would) -- this must
        # not be double-counted as an independent, conflicting positive
        # statement against its own section's negative answer.
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section("Timer", "This toothbrush does not have a timer.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "philips_p1", warnings)
        timer_av, desc_av = utils.extract_timer(sections_, "philips_p1", record, store, warnings)
        assert timer_av.status == "stated_absent"
        assert timer_av.value is False
        assert desc_av.status == "not_mentioned"

    def test_not_mentioned_when_topic_never_appears(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section("Intro", "A nice toothbrush for everyone.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "philips_p1", warnings)
        pressure_av = utils.extract_pressure_sensing(sections_, "philips_p1", record, store, warnings)
        assert pressure_av.status == "not_mentioned"
        assert pressure_av.value is None
        assert pressure_av.evidence_ids == []

    def test_stated_with_unnormalizable_value_for_qualitative_battery(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section("Battery", "Enjoy long-lasting battery performance.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "philips_p1", warnings)
        days_av, raw_av = utils.extract_battery_life(sections_, "philips_p1", record, store, warnings)
        assert days_av.status == "stated"
        assert days_av.value is None
        assert raw_av.status == "stated"
        assert raw_av.value is not None
        assert any(w.code == "unnormalizable_value" and w.attribute == "battery_life_days" for w in warnings)

    def test_genuine_conflict_is_ambiguous_for_mode_count(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[
                section("Modes A", "This toothbrush has 5 modes."),
                section("Modes B", "Actually it has 7 modes available."),
            ],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "oralb_o1", warnings)
        mode_count_av, _ = utils.extract_modes(sections_, "oralb_o1", record, store, warnings)
        assert mode_count_av.status == "ambiguous"
        assert mode_count_av.value is None
        assert len(mode_count_av.evidence_ids) == 2
        assert any(w.code == "mode_count_conflict" for w in warnings)

    def test_numeric_battery_life_normalizes_weeks_to_days(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section("Battery", "Battery life lasts up to 2 weeks on a single charge.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "philips_p1", warnings)
        days_av, raw_av = utils.extract_battery_life(sections_, "philips_p1", record, store, warnings)
        assert days_av.status == "stated"
        assert days_av.value == 14
        assert "2 weeks" in raw_av.value


# ---------------------------------------------------------------------------
# App vs Bluetooth
# ---------------------------------------------------------------------------

class TestAppVsBluetooth:
    def test_bluetooth_mention_does_not_imply_app_connectivity(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section("Features", "Bluetooth® Connectivity included for tracking.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "oralb_o1", warnings)
        app_av = utils.extract_app_connectivity(sections_, "oralb_o1", record, store, warnings)
        bt_av = utils.extract_bluetooth_connectivity(sections_, "oralb_o1", record, store, warnings)
        assert bt_av.status == "stated" and bt_av.value is True
        assert app_av.status == "not_mentioned" and app_av.value is None

    def test_explicit_app_mention_sets_app_connectivity_independently(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section("Features", "Connects to the Oral-B app for personalized coaching.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "oralb_o1", warnings)
        app_av = utils.extract_app_connectivity(sections_, "oralb_o1", record, store, warnings)
        assert app_av.status == "stated" and app_av.value is True


# ---------------------------------------------------------------------------
# Charging method: magnetic charger vs magnetic drive motor
# ---------------------------------------------------------------------------

class TestChargingMethod:
    def test_magnetic_charger_is_recognized(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section("Specs", "Comes with a Magnetic Charger for fast charging.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "oralb_o1", warnings)
        method_av, raw_av = utils.extract_charging_method(sections_, "oralb_o1", record, store, warnings)
        assert method_av.value == "magnetic"
        assert method_av.status == "stated"

    def test_magnetic_drive_motor_is_not_confused_with_a_charging_method(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section("Specs", "Powered by a magnetic drive motor for smooth vibrations.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "oralb_o1", warnings)
        method_av, _ = utils.extract_charging_method(sections_, "oralb_o1", record, store, warnings)
        assert method_av.status == "not_mentioned"
        assert method_av.value is None

    def test_unrecognized_charging_terminology_is_stated_but_unnormalizable(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section("Specs", "Comes with a Charging Station for easy use.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "oralb_o1", warnings)
        method_av, raw_av = utils.extract_charging_method(sections_, "oralb_o1", record, store, warnings)
        assert method_av.status == "stated"
        assert method_av.value is None
        assert raw_av.value == "Charging Station"
        assert any(w.code == "unnormalizable_value" and w.attribute == "charging_method" for w in warnings)


# ---------------------------------------------------------------------------
# Pressure sensing
# ---------------------------------------------------------------------------

class TestPressureSensing:
    def test_explicit_pressure_sensor_is_stated(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section("Care", "Built-in pressure sensor alerts you when brushing too hard.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "philips_p1", warnings)
        av = utils.extract_pressure_sensing(sections_, "philips_p1", record, store, warnings)
        assert av.status == "stated" and av.value is True

    def test_not_inferred_from_product_tier_or_marketing_fluff(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section("Premium Flagship", "Our most advanced premium toothbrush ever.")],
        )
        store, warnings = EvidenceStore(), []
        sections_ = usable_sections(record["sections"], "philips_p1", warnings)
        av = utils.extract_pressure_sensing(sections_, "philips_p1", record, store, warnings)
        assert av.status == "not_mentioned"


# ---------------------------------------------------------------------------
# Cross-sell / comparison-FAQ exclusion (filters.usable_sections)
# ---------------------------------------------------------------------------

class TestSectionFiltering:
    def test_cross_sell_section_is_excluded(self):
        sections = [
            section("You May Also Like", "Other Product $99.99"),
            section("Real Content", "Pressure sensor included."),
        ]
        warnings = []
        usable = usable_sections(sections, "p1", warnings)
        headings = [s["heading"] for _, s in usable]
        assert "You May Also Like" not in headings
        assert "Real Content" in headings

    def test_comparison_faq_is_excluded_and_warned(self):
        sections = [section("What's the difference between Series 9 and 10?", "Series 9 has 4 heads; Series 10 has 2.")]
        warnings = []
        usable = usable_sections(sections, "oralb_series10", warnings)
        assert usable == []
        assert len(warnings) == 1
        assert warnings[0].code == "cross_product_comparison_section_excluded"

    def test_reviews_section_excluded_without_a_warning(self):
        sections = [section("Reviews", "I love the app!")]
        warnings = []
        usable = usable_sections(sections, "p1", warnings)
        assert usable == []
        assert warnings == []

    def test_cross_sell_content_never_leaks_into_another_products_attributes(self):
        sections = [
            section("You May Also Like", "The Series 5 has a pressure sensor and costs $99.99."),
            section("Real Content", "A great toothbrush for daily use."),
        ]
        warnings = []
        usable = usable_sections(sections, "p1", warnings)
        store = EvidenceStore()
        fake_record = {"product_url": "https://example.com/p1", "raw_html_path": "p1.html"}
        pressure_av = utils.extract_pressure_sensing(usable, "p1", fake_record, store, warnings)
        assert pressure_av.status == "not_mentioned"


# ---------------------------------------------------------------------------
# Product IDs and variants
# ---------------------------------------------------------------------------

class TestProductIdAndVariants:
    def test_product_id_matches_raw_html_filename_stem(self, tmp_path):
        record = make_raw_record(tmp_path, brand="Philips", slug="hx9990-13", json_ld_price=379.96)
        store, warnings = EvidenceStore(), []
        product = philips.normalize_product(record, store, warnings)
        assert product.product_id == "philips_hx9990-13"

    def test_product_id_is_deterministic_across_repeated_runs(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="io-series-9-rechargeable-electric-toothbrush",
            json_ld_price="169.99", price_text="$169.99",
        )
        p1 = oralb.normalize_product(record, EvidenceStore(), [])
        p2 = oralb.normalize_product(record, EvidenceStore(), [])
        assert p1.product_id == p2.product_id == "oralb_io-series-9-rechargeable-electric-toothbrush"

    def test_product_id_uses_model_slug_not_color_specific_sku(self, tmp_path):
        # Oral-B's JSON-LD sku is per-color; the product_id must come from
        # the Step 1 filename (the stable, model-level slug) instead.
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="io-series-9-rechargeable-electric-toothbrush",
            json_ld_price="169.99",
            extra_json_ld_blocks=[],
        )
        record["structured_data"]["json_ld"][0]["sku"] = "80338587"  # a color-specific variant SKU
        product = oralb.normalize_product(record, EvidenceStore(), [])
        assert product.product_id == "oralb_io-series-9-rechargeable-electric-toothbrush"
        assert "80338587" not in product.product_id

    def test_variants_are_preserved_without_creating_extra_products(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="io-series-9-rechargeable-electric-toothbrush",
            json_ld_price="169.99", price_text="$169.99",
            variants=[
                {"url": "https://oralb.com/.../black-onyx/", "label": "Black Onyx", "sku": None},
                {"url": "https://oralb.com/.../rose-quartz/", "label": "Rose Quartz", "sku": None},
            ],
        )
        product = oralb.normalize_product(record, EvidenceStore(), [])
        assert len(product.variants) == 2
        assert {v.label for v in product.variants} == {"Black Onyx", "Rose Quartz"}


# ---------------------------------------------------------------------------
# CSV serialization
# ---------------------------------------------------------------------------

class TestCsvSerialization:
    def test_csv_row_count_equals_normalized_product_count(self, tmp_path):
        import build_table

        record1 = make_raw_record(tmp_path, brand="Philips", slug="p1", json_ld_price=19.99)
        record2 = make_raw_record(tmp_path, brand="Oral-B", slug="o1", json_ld_price="29.99", price_text="$29.99")
        products = [
            philips.normalize_product(record1, EvidenceStore(), []),
            oralb.normalize_product(record2, EvidenceStore(), []),
        ]
        csv_path = tmp_path / "out.csv"
        build_table.write_csv(products, csv_path)

        import csv as csv_module
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv_module.DictReader(f))
        assert len(rows) == len(products) == 2

    def test_not_mentioned_renders_as_empty_cell_not_a_string(self, tmp_path):
        import build_table

        record = make_raw_record(tmp_path, brand="Philips", slug="p1", json_ld_price=19.99)
        product = philips.normalize_product(record, EvidenceStore(), [])
        csv_path = tmp_path / "out.csv"
        build_table.write_csv([product], csv_path)

        import csv as csv_module
        with open(csv_path, encoding="utf-8") as f:
            row = next(csv_module.DictReader(f))
        assert row["pressure_sensing"] == ""  # not "not_mentioned", not "None", not "null"
        assert row["app_connectivity"] == ""

    def test_list_fields_are_json_encoded(self, tmp_path):
        import build_table
        import json as json_module

        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1", json_ld_price="29.99", price_text="$29.99",
            sections=[section("Specs", "1 Oral-B iO Gentle Care Clean Replacement Brush Head")],
        )
        product = oralb.normalize_product(record, EvidenceStore(), [])
        csv_path = tmp_path / "out.csv"
        build_table.write_csv([product], csv_path)

        import csv as csv_module
        with open(csv_path, encoding="utf-8") as f:
            row = next(csv_module.DictReader(f))
        parsed = json_module.loads(row["included_items"])
        assert isinstance(parsed, list)
        assert parsed[0] == "1 Oral-B iO Gentle Care Clean Replacement Brush Head"


# ---------------------------------------------------------------------------
# Integration: the real Step 1 output, end to end
# ---------------------------------------------------------------------------

class TestRealDataIntegration:
    def test_full_pipeline_on_real_raw_data_passes_validation(self):
        root = Path(__file__).resolve().parents[1]
        raw_dir = root / "data" / "raw"
        records = load_raw_records(raw_dir)
        if not records:
            import pytest

            pytest.skip("data/raw/ is empty in this environment; run scripts/build_data.py first")

        products, store, _warnings = normalize_all(records)
        raw_index = index_raw_records(records)
        errors = validate(products, store, raw_index)
        assert errors == [], f"validation errors against real data: {errors}"

        product_ids = [p.product_id for p in products]
        assert len(product_ids) == len(set(product_ids)), "duplicate product_id in real data"
