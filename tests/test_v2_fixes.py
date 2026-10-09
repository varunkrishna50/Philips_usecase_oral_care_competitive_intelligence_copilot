"""Regression tests for review round 2 (Feedback v2.txt), run against the
real stored snapshots. Each test names the feedback point it covers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acquisition.utils import clean_soup, extract_sections  # noqa: E402
from normalization.normalize import load_raw_records, normalize_all  # noqa: E402
from normalization.validation import validate  # noqa: E402
from normalization.normalize import index_raw_records  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def built():
    records = [r for r in load_raw_records(RAW_DIR) if r.get("details")]
    if not records or not Path(records[0]["raw_html_path"]).exists():
        pytest.skip("raw records / HTML snapshots not available (run from repo root)")
    products, store, warnings = normalize_all(records)
    return {p.product_id: p for p in products}, store, records, warnings


def _p(built, fragment):
    products = built[0]
    matches = [p for pid, p in products.items() if fragment in pid]
    if not matches:
        pytest.skip(f"{fragment} not in the current catalog")
    return matches[0]


def test_validation_passes_with_all_extras(built):
    products, store, records, _ = built
    assert validate(list(products.values()), store, index_raw_records(records)) == []


# --- section extraction (root cause 1) -----------------------------------------------------------

def test_bare_text_in_ul_and_br_split_are_kept():
    html = "<body><h2>Q</h2><div><ul>Clock: a<li>x</li>Timer: b</ul></div><p>one<br>two</p></body>"
    text = "\n".join(s["text"] for s in extract_sections(clean_soup(html)))
    assert "Timer: b" in text and "Clock: a" in text
    assert "one\ntwo" in text


# --- Oral-B Series 10 -----------------------------------------------------------------------------------

def test_s10_out_of_stock_hides_price_and_states_availability(built):
    p = _p(built, "io-series-10")
    assert p.availability.value == "Out of stock"
    assert p.price.status == "not_mentioned"
    assert p.listed_price_hidden.value == 399.99


def test_s10_timer_includes_clock_and_brushing_timer(built):
    p = _p(built, "io-series-10")
    assert "Clock Display" in p.timer_description.value and "Brushing Timer" in p.timer_description.value


def test_s10_details_charge_time_zones_and_plaque_claim(built):
    p = _p(built, "io-series-10")
    assert p.charge_time_hours.value == 3.0
    assert (p.brushing_zones.value, p.zone_interval_seconds.value, p.noise_db.value) == (6, 20, 64)
    assert p.plaque_removal_claim.value.endswith("vs. a manual brush.")
    assert any("Live coaching" in s for s in p.app_connectivity_details.value)
    assert any("too much force" in s for s in p.pressure_sensing_details.value)
    assert all("consent" not in s for s in p.pressure_sensing_details.value)


# --- Oral-B Series 2 -------------------------------------------------------------------------------------

def test_s2_description_is_the_visible_lead_paragraph(built):
    p = _p(built, "io-series-2")
    assert p.description.value.startswith("The Oral-B iO Series 2 combines powerful yet gentle micro-vibrations")


def test_s2_own_side_comparison_facts_and_battery(built):
    p = _p(built, "io-series-2")
    assert "micro-vibrations" in p.cleaning_technology.value
    assert "iO specific replacement brush heads" in p.brush_head_compatibility.value
    assert p.feel_and_noise.value.startswith("Smoother and quieter")
    assert "Entry point" in p.price_positioning.value
    assert any("low-charge indicator" in s for s in p.battery_details.value)
    assert p.brush_heads_included.status == "not_mentioned"  # quantity not stated


# --- Oral-B Series 3 ---------------------------------------------------------------------------------------

def test_s3_claims_are_single_sentences_and_display_is_stated(built):
    p = _p(built, "io-series-3")
    assert all(len(c) < 300 for c in p.claims.value)
    assert any("400% more plaque" in c for c in p.claims.value)
    assert "light ring" in p.display_type.value
    assert "every 3 months" in p.brush_head_replacement_interval.value
    # the warranty-tab footnote is not a disclaimer of any claim
    assert p.disclaimers.status == "not_mentioned"


# --- Oral-B Series 4 / 5 -------------------------------------------------------------------------------------

@pytest.mark.parametrize("fragment", ["io-series-4-rechargeable", "io-series-5-customizable"])
def test_modes_equal_structured_mode_names(built, fragment):
    p = _p(built, fragment)
    assert p.modes.value == [m["name"] for m in p.mode_details.value]


def test_s4_description_is_complete_and_footnotes_merged(built):
    p = _p(built, "io-series-4-rechargeable")
    assert p.description.value.endswith(".")
    markers = [d["marker"] for d in p.disclaimers.value]
    assert markers.count("*") == 1


def test_s5_timer_includes_30_second_vibration(built):
    p = _p(built, "io-series-5")
    assert "vibrates every 30 seconds" in p.timer_description.value
    assert p.replacement_reminder.value is True


# --- Philips hx4031-71 ------------------------------------------------------------------------------------------

def test_philips_hx4031_specs(built):
    p = _p(built, "philips_hx4031-71")
    assert p.battery_life_days.value == 14
    assert p.charging_method.value == "charger"
    assert p.included_items.value == [
        "Handle: 1 3100 rechargeable toothbrush", "Brush head: 1 Optimal Plaque C2", "Charger: 1",
    ]
    assert p.brush_movements_per_min.value == 31000
    assert p.intensity_count.value == 3 and p.intensity_levels.value == ["High", "Medium", "Low"]
    assert {"name": "High", "description": "To boost your clean"} in p.intensity_details.value
    assert p.battery_indicator.value.startswith("Illuminated icon")
    assert any(s.startswith("Technical specifications: Battery type - Lithium") for s in p.specifications.value)
    assert any("haptic feedback" in s for s in p.pressure_sensing_details.value)
    assert "BrushPacer" in p.timer_description.value


def test_philips_disclaimer_locations_are_labelled_not_quoted(built):
    p = _p(built, "philips_hx4031-71")
    where = [a["where"] for d in p.disclaimers.value for a in d["applies_to"]]
    assert "Feature panel 'Say goodbye to plaque'" in where
    assert not any(w.startswith("Feature: ") for w in where)
