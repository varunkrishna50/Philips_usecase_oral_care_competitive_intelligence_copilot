"""Regression tests for the targeted extraction corrections:

1. FAQ questions must never be used as the evidence for their own answer.
2. included_items must pick up bare accessory names from the labeled
   TextFeatureN/"Shop By Feature" attribute pairs, not just digit-
   prefixed lines.
3. Claims must never be contaminated by a glued heading/tab-label
   prefix, and duplicates (with or without that contamination) must
   collapse to one.
4. battery_life must not fire on a "Battery Toothbrush" product name or
   a battery-level-indicator statement, but must parse "1 charge lasts
   N weeks" style runtime statements.
5. Intensity settings must never be converted into mode_count.
6. Philips "N brush mode(s), M intensities" phrasing must populate
   mode_count correctly.
7. description must not be a specification/feature label.

Each fix also gets a real-data check against the actual Step 1 output
for the specific product named when the task that introduced the fix
called it out, so a future change can't silently regress it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from normalization import oralb, philips, utils
from normalization.evidence import EvidenceStore
from normalization.filters import usable_sections

from normalization_fixtures import make_raw_record, section

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _load_real_record(brand_dir: str, slug: str) -> dict:
    path = RAW_DIR / brand_dir / f"{slug}.json"
    if not path.exists():
        pytest.skip(f"{path} not present in this environment; run scripts/build_data.py first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. FAQ question must never be used as evidence
# ---------------------------------------------------------------------------

class TestFaqQuestionNotUsedAsEvidence:
    def test_question_containing_timer_does_not_set_timer_from_the_question_alone(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section(
                "Does the Oral-B iO Series 10 charger display the time and a brushing timer?",
                "Yes, the iO Sense charger displays both a clock and a brushing timer.",
            )],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        timer_av, desc_av = utils.extract_timer(usable, "oralb_o1", record, EvidenceStore(), [])
        assert timer_av.status == "stated"
        # The evidence must be the ANSWER, never the question itself.
        assert desc_av.value == "Yes, the iO Sense charger displays both a clock and a brushing timer."
        assert "?" not in desc_av.value

    def test_question_containing_battery_does_not_create_battery_life_from_the_question_alone(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section(
                "How long does an Oral-B iO Series 9 battery last?",
                "The Oral-B iO Series 9 uses a long-lasting rechargeable lithium-ion battery.",
            )],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        days_av, raw_av = utils.extract_battery_life(usable, "oralb_o1", record, EvidenceStore(), [])
        assert raw_av.status == "stated"
        assert raw_av.value == "The Oral-B iO Series 9 uses a long-lasting rechargeable lithium-ion battery."
        assert "?" not in raw_av.value

    def test_answer_clearly_associated_with_its_question_is_still_usable(self, tmp_path):
        """A question-framed heading doesn't disqualify the real answer
        text that follows it in the same section -- only the question
        wording itself is excluded from being treated as the evidence."""
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section(
                "Does the Oral-B iO Series 2 have a pressure sensor?",
                "Yes, it has an automatic gum pressure sensor that signals when you brush too hard.",
            )],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        store = EvidenceStore()
        av = utils.extract_pressure_sensing(usable, "oralb_o1", record, store, [])
        assert av.status == "stated"
        assert av.value is True
        evidence_text = store.get(av.evidence_ids[0]).evidence_text
        assert "automatic gum pressure sensor" in evidence_text
        assert "?" not in evidence_text

    def test_real_data_series_10_timer_description_is_the_answer(self):
        record = _load_real_record("oralb", "oralb_io-series-10-rechargeable-electric-toothbrush")
        store, warnings = EvidenceStore(), []
        product = oralb.normalize_product(record, store, warnings)
        assert product.timer_description.value is not None
        assert "?" not in product.timer_description.value
        assert "iO Sense" in product.timer_description.value

    def test_real_data_series_9_battery_raw_is_not_the_question(self):
        record = _load_real_record("oralb", "oralb_io-series-9-rechargeable-electric-toothbrush")
        store, warnings = EvidenceStore(), []
        product = oralb.normalize_product(record, store, warnings)
        assert product.battery_life_raw.value is not None
        assert "?" not in product.battery_life_raw.value


# ---------------------------------------------------------------------------
# 2. included_items from labeled feature pairs (bare accessory names)
# ---------------------------------------------------------------------------

class TestIncludedItemsFromBoxList:
    """Oral-B included_items = the page's "What's in the box" list. Internal
    merchandising fields (TextFeatureN / Shop By Feature) are never evidence."""

    def test_series2_box_list_is_exact_and_has_no_travel_case(self):
        record = _load_real_record("oralb", "oralb_io-series-2-rechargeable-electric-toothbrush-starter-kit")
        store, warnings = EvidenceStore(), []
        product = oralb.normalize_product(record, store, warnings)
        assert product.included_items.value == [
            "Oral-B iO2 Electric Toothbrush",
            "Gum & Sensitive Replacement Brush Head",
            "Ultimate Clean Replacement Brush Head",
            "Charger",
        ]
        assert product.travel_case_included.status == "not_mentioned"
        assert product.brush_heads_included.status == "not_mentioned"  # two head types, no quantity stated

    def test_hidden_custom_fields_are_never_evidence(self):
        for slug in ("oralb_io-series-2-rechargeable-electric-toothbrush-starter-kit",
                     "oralb_io-series-9-rechargeable-electric-toothbrush"):
            record = _load_real_record("oralb", slug)
            store = EvidenceStore()
            oralb.normalize_product(record, store, [])
            for ev in store.all().values():
                assert not ev.evidence_text.startswith(("TextFeature", "Shop By Feature", "algolia", "bvProductId")), ev
            assert "Shop By Feature" not in record["raw_text"]

    @pytest.mark.parametrize(
        "slug",
        [
            "oralb_io-series-2-rechargeable-electric-toothbrush-starter-kit",
            "oralb_io-series-4-rechargeable-electric-toothbrush",
            "oralb_io-series-5-customizable-clean-rechargeable-electric-toothbrush",
            "oralb_io-series-7-electric-toothbrush",
            "oralb_io-series-8-electric-toothbrush",
            "oralb_io-series-9-rechargeable-electric-toothbrush",
            "oralb_io-series-10-rechargeable-electric-toothbrush",
        ],
    )
    def test_real_data_included_items_go_beyond_just_brush_heads(self, slug):
        record = _load_real_record("oralb", slug)
        store, warnings = EvidenceStore(), []
        product = oralb.normalize_product(record, store, warnings)
        assert product.included_items.status == "stated"
        values = " ".join(product.included_items.value or [])
        assert "brush" in values.lower()  # the handle itself or a brush head
        # At least one non-brush-head package component (charger/case/pouch).
        assert any(
            kw in values.lower() for kw in ("charger", "case", "pouch", "station")
        ), f"{slug}: included_items has no charging/case component: {values!r}"


# ---------------------------------------------------------------------------
# 3. Claims: no heading contamination, dedup, no reconstruction
# ---------------------------------------------------------------------------

class TestClaimsContamination:
    def test_tab_label_prefix_is_stripped(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section(
                "Main",
                "What's in the box The iO Series 2 removes 100% more plaque even in hard-to-reach areas.",
            )],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        av = utils.extract_claims(usable, "oralb_o1", record, EvidenceStore(), [])
        assert av.status == "stated"
        assert av.value == ["The iO Series 2 removes 100% more plaque even in hard-to-reach areas."]

    def test_duplicate_claims_with_different_glued_prefixes_collapse_to_one(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section(
                "Main",
                "The iO Series 2 removes 100% more plaque in hard-to-reach areas.\n"
                "What's in the box The iO Series 2 removes 100% more plaque in hard-to-reach areas.\n"
                "Featured in The iO Series 2 removes 100% more plaque in hard-to-reach areas.\n"
                "Warranty Information The iO Series 2 removes 100% more plaque in hard-to-reach areas.\n",
            )],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        av = utils.extract_claims(usable, "oralb_o1", record, EvidenceStore(), [])
        assert av.value == ["The iO Series 2 removes 100% more plaque in hard-to-reach areas."]

    def test_no_incomplete_claim_is_reconstructed_with_a_guess(self, tmp_path):
        """A fragment ending mid-sentence must be kept verbatim (including
        its trailing "vs.") or dropped -- never completed with invented
        text naming a specific competitor/product."""
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section("Main", "The iO Series 2 removes 100% more plaque even in hard-to-reach areas vs.")],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        av = utils.extract_claims(usable, "oralb_o1", record, EvidenceStore(), [])
        assert av.value == ["The iO Series 2 removes 100% more plaque even in hard-to-reach areas vs."]
        assert "manual toothbrush" not in av.value[0]  # nothing invented past the real fragment

    def test_review_section_claims_are_excluded(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section("Reviews", "I love the app, removes 7x more plaque than my old brush!")],
        )
        usable = usable_sections(record["sections"], "p1", [])
        assert usable == []  # Reviews is excluded before claims ever sees it
        av = utils.extract_claims(usable, "p1", record, EvidenceStore(), [])
        assert av.status == "not_mentioned"

    def test_evidence_text_is_an_exact_substring_of_the_section(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section("Main", "Removes up to 7x more plaque than a manual toothbrush.")],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        store = EvidenceStore()
        av = utils.extract_claims(usable, "oralb_o1", record, store, [])
        for eid in av.evidence_ids:
            ev = store.get(eid)
            assert ev.evidence_text in record["sections"][0]["heading"] + "\n" + record["sections"][0]["text"]

    def test_real_data_series_2_claims_have_no_glued_prefix_and_no_duplicates(self):
        record = _load_real_record("oralb", "oralb_io-series-2-rechargeable-electric-toothbrush-starter-kit")
        store, warnings = EvidenceStore(), []
        product = oralb.normalize_product(record, store, warnings)
        claims = product.claims.value or []
        assert len(claims) == len(set(claims))  # no duplicates
        for c in claims:
            assert not c.startswith("What's in the box ")
            assert not c.startswith("Make the most of every session ")
            assert not c.startswith("Featured in ")
            assert not c.startswith("Warranty Information ")
            assert not c.startswith("More Information ")


# ---------------------------------------------------------------------------
# 4. battery_life false positives
# ---------------------------------------------------------------------------

class TestBatteryLifeFalsePositives:
    def test_battery_toothbrush_product_title_is_not_battery_life_evidence(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1", product_name="Philips One by Sonicare Battery Toothbrush",
            sections=[section("Philips One by Sonicare Battery Toothbrush", "Good vibrations and a simple clean.")],
        )
        usable = usable_sections(record["sections"], "p1", [])
        days_av, raw_av = utils.extract_battery_life(usable, "p1", record, EvidenceStore(), [])
        assert days_av.status == "not_mentioned"
        assert raw_av.status == "not_mentioned"

    def test_battery_level_indicator_statement_is_not_battery_life(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section(
                "iO Series 10",
                "The display always tracks your battery level so you know when it's time to recharge.",
            )],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        days_av, raw_av = utils.extract_battery_life(usable, "oralb_o1", record, EvidenceStore(), [])
        assert days_av.status == "not_mentioned"
        assert raw_av.status == "not_mentioned"

    def test_charge_lasts_n_weeks_is_parsed_as_runtime(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section("Battery", "Long-lasting battery:\n1 charge lasts 4 weeks, making it great for travel.")],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        days_av, raw_av = utils.extract_battery_life(usable, "oralb_o1", record, EvidenceStore(), [])
        assert days_av.status == "stated"
        assert days_av.value == 28

    def test_real_data_one_by_sonicare_battery_toothbrush_not_mentioned(self):
        record = _load_real_record("philips", "philips_hy1100-02")
        store, warnings = EvidenceStore(), []
        product = philips.normalize_product(record, store, warnings)
        assert product.battery_life_days.status == "not_mentioned"
        assert product.battery_life_raw.status == "not_mentioned"

    def test_real_data_one_for_kids_battery_life_comes_from_the_spec_row_not_the_product_name(self):
        """The product *name* contains "Battery toothbrush" (a power-source
        descriptor, never a runtime). The runtime is stated in the spec
        table row "Operating time (full to empty): 90 days"."""
        record = _load_real_record("philips", "philips_hy1130-02")
        store, warnings = EvidenceStore(), []
        product = philips.normalize_product(record, store, warnings)
        assert product.battery_life_days.value == 90
        assert product.battery_life_raw.value == "Operating time (full to empty): 90 days"

    def test_real_data_series_10_battery_level_indicator_excluded(self):
        record = _load_real_record("oralb", "oralb_io-series-10-rechargeable-electric-toothbrush")
        store, warnings = EvidenceStore(), []
        product = oralb.normalize_product(record, store, warnings)
        if product.battery_life_raw.value is not None:
            assert "tracks your battery level" not in product.battery_life_raw.value.lower()


# ---------------------------------------------------------------------------
# 5. Mode count vs intensity count
# ---------------------------------------------------------------------------

class TestModeVsIntensityCount:
    def test_intensity_settings_alone_do_not_populate_mode_count(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section(
                "2 intensity settings for a more powerful or gentle clean",
                "Switch between 2 intensity settings to make the brushing session comfortable for you.",
            )],
        )
        usable = usable_sections(record["sections"], "p1", [])
        mode_count_av, _ = utils.extract_modes(usable, "p1", record, EvidenceStore(), [])
        assert mode_count_av.status == "not_mentioned"

    def test_explicit_mode_count_alongside_intensities_is_used(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="p1",
            sections=[section("Features", "3 brush modes, 3 intensities")],
        )
        usable = usable_sections(record["sections"], "p1", [])
        mode_count_av, _ = utils.extract_modes(usable, "p1", record, EvidenceStore(), [])
        assert mode_count_av.status == "stated"
        assert mode_count_av.value == 3

    def test_real_data_hx3681_03_mode_count_is_null(self):
        record = _load_real_record("philips", "philips_hx3681-03")
        store, warnings = EvidenceStore(), []
        product = philips.normalize_product(record, store, warnings)
        assert product.mode_count.status == "not_mentioned"


# ---------------------------------------------------------------------------
# 6. Philips mode extraction ("brush mode(s)")
# ---------------------------------------------------------------------------

class TestPhilipsModeExtraction:
    def test_hx7130_01_two_brush_mode_singular(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="hx7130-01",
            sections=[section("Features", "2 brush mode, 3 intensities")],
        )
        usable = usable_sections(record["sections"], "philips_hx7130-01", [])
        mode_count_av, _ = utils.extract_modes(usable, "philips_hx7130-01", record, EvidenceStore(), [])
        assert mode_count_av.status == "stated"
        assert mode_count_av.value == 2

    def test_hx7410_10_three_brush_modes_plural(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Philips", slug="hx7410-10",
            sections=[section("Features", "3 brush modes, 3 intensities")],
        )
        usable = usable_sections(record["sections"], "philips_hx7410-10", [])
        mode_count_av, _ = utils.extract_modes(usable, "philips_hx7410-10", record, EvidenceStore(), [])
        assert mode_count_av.status == "stated"
        assert mode_count_av.value == 3

    def test_real_data_hx7130_01_mode_count_is_2(self):
        record = _load_real_record("philips", "philips_hx7130-01")
        store, warnings = EvidenceStore(), []
        product = philips.normalize_product(record, store, warnings)
        assert product.mode_count.status == "stated"
        assert product.mode_count.value == 2

    def test_real_data_hx7410_10_mode_count_is_3(self):
        record = _load_real_record("philips", "philips_hx7410-10")
        store, warnings = EvidenceStore(), []
        product = philips.normalize_product(record, store, warnings)
        assert product.mode_count.status == "stated"
        assert product.mode_count.value == 3


# ---------------------------------------------------------------------------
# 7. Description must not be a specification label
# ---------------------------------------------------------------------------

class TestDescriptionNotASpecLabel:
    def test_short_spec_bullet_is_skipped_in_favor_of_real_sentence(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section(
                "Main",
                "3 SMART Brushing Modes\n"
                "Smart Pressure Sensor\n"
                "The iO Series 2 removes 100% more plaque even in hard-to-reach areas vs. a regular manual toothbrush.",
            )],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        av = utils.extract_description(usable, "oralb_o1", record, EvidenceStore(), [])
        assert av.status == "stated"
        assert av.value.startswith("The iO Series 2 removes")

    def test_internal_identifier_is_never_used_as_description(self, tmp_path):
        record = make_raw_record(
            tmp_path, brand="Oral-B", slug="o1",
            sections=[section(
                "Main",
                "lightbox-inline-form-8f6c3d90-7060-48be-a392-d9670fd8f787\n"
                "The Oral-B iO Series 4 features our best technology for a professional, purifying clean.",
            )],
        )
        usable = usable_sections(record["sections"], "oralb_o1", [])
        av = utils.extract_description(usable, "oralb_o1", record, EvidenceStore(), [])
        assert "lightbox" not in (av.value or "")

    def test_real_data_series_2_description_is_prose_not_a_spec_tag(self):
        record = _load_real_record(
            "oralb", "oralb_io-series-2-clean-protect-rechargeable-electric-toothbrush-starter-kit"
        )
        store, warnings = EvidenceStore(), []
        product = oralb.normalize_product(record, store, warnings)
        assert product.description.status == "stated"
        assert "Brushing Modes" not in product.description.value
        assert len(product.description.value) > 50
