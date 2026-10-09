"""Tests for structured product-detail extraction (prices, colors, modes,
features, ratings). Page-level tests run against the HTML snapshots that
Step 1 stores under data/raw/html and are skipped if those are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acquisition import details as d  # noqa: E402
from acquisition import oralb, philips  # noqa: E402
from acquisition.utils import extract_json_ld  # noqa: E402

HTML_DIR = ROOT / "data" / "raw" / "html"


def _load(brand: str, name: str):
    path = HTML_DIR / brand / name
    if not path.exists():
        pytest.skip(f"{path} not available")
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    ld = [b for b in extract_json_ld(soup) if b.get("@type") == "Product"]
    return html, soup, ld


# --- pure helpers --------------------------------------------------------

def test_compute_discount():
    assert d.compute_discount(329.99, 169.99) == (160.0, 48.5)
    assert d.compute_discount(19.96, 19.96) == (0.0, 0.0)
    assert d.compute_discount(None, 10.0) == (None, None)


def test_parse_mode_list_splits_names_and_descriptions():
    line = "Daily Clean for everyday teeth and gums, Whiten to make teeth brighter, and Tongue Clean to freshen."
    modes = d.parse_mode_list(line)
    assert [m.name for m in modes] == ["Daily Clean", "Whiten", "Tongue Clean"]
    assert modes[0].description == "for everyday teeth and gums"
    assert modes[1].description == "to make teeth brighter"


def test_extract_modes_prefers_list_matching_stated_count():
    lines = [
        "Personalize your clean with 3 Smart Brushing Modes : Daily Clean, Sensitive, and Whitening.",
        "Something unrelated.",
    ]
    count, modes = d.extract_modes_from_lines(lines)
    assert count == 3
    assert [m.name for m in modes] == ["Daily Clean", "Sensitive", "Whitening"]


def test_html_to_lines_ignores_soft_wraps():
    lines = d.html_to_lines("<li>healthy teeth and\n   gums</li><li>second</li>")
    assert lines == ["healthy teeth and gums", "second"]


# --- Philips -------------------------------------------------------------

def test_philips_prestige_9900_details():
    html, soup, ld = _load("philips", "philips_hx9990-13.html")
    det = philips._extract_details(html, soup, "HX9990_13", ld)
    assert (det.original_price, det.sale_price, det.currency) == (399.96, 379.96, "USD")
    assert det.discount_amount == 20.0 and det.discount_pct == 5.0
    assert det.color_count == 3 and "Midnight Blue" in det.colors
    assert det.mode_count == 5
    assert [m.name for m in det.modes] == ["Clean", "Deep Clean", "Sensitive", "Gum Health", "White+"]
    assert all(m.description for m in det.modes)
    assert det.intensities == ["High", "Medium", "Low"]
    assert any(f.name == "SenseIQ Technology" for f in det.features)
    assert (det.rating_avg, det.rating_count) == (4.2, 488)
    assert not any(n.startswith(("mode_count_mismatch", "json_ld_price_differs")) for n in det.notes)


def test_philips_no_discount_means_zero_percent():
    html, soup, ld = _load("philips", "philips_hx7410-10.html")
    det = philips._extract_details(html, soup, "HX7410_10", ld)
    assert det.original_price == det.sale_price == 129.96
    assert det.discount_pct == 0.0


def test_philips_single_mode():
    html, soup, ld = _load("philips", "philips_hx3681-03.html")
    det = philips._extract_details(html, soup, "HX3681_03", ld)
    assert det.mode_count == 1 and det.modes[0].name == "Clean"


# --- Oral-B --------------------------------------------------------------

def test_oralb_io9_details():
    html, soup, ld = _load("oralb", "oralb_io-series-9-rechargeable-electric-toothbrush.html")
    det = oralb._extract_details(html, soup, ld, None, [])
    # Promotions come and go (iO 9 was $169.99 until early Oct 2026): assert the
    # invariants, not a particular promo state.
    assert det.original_price == 329.99 and det.currency == "USD"
    assert det.sale_price <= det.original_price
    assert round(det.original_price - det.sale_price, 2) == det.discount_amount
    assert det.mode_count == 7
    assert [m.name for m in det.modes] == [
        "Daily Clean", "Whiten", "Gum Care", "Sensitive", "Super Sensitive", "Intense Clean", "Tongue Clean",
    ]
    assert det.modes[1].description == "to make your teeth brighter"
    assert det.color_count == len(det.colors) >= 4
    assert "interactive color display" in [f.name.lower() for f in det.features]
    assert not any(n.startswith(("mode_count_mismatch", "saved_amount_mismatch")) for n in det.notes)


def test_oralb_full_price_product_has_no_discount():
    html, soup, ld = _load("oralb", "oralb_io-series-7-electric-toothbrush.html")
    det = oralb._extract_details(html, soup, ld, None, [])
    assert det.original_price == det.sale_price == 219.99
    assert det.discount_pct == 0.0
    assert det.mode_count == 5 and len(det.modes) == 5


def test_oralb_series2_modes_come_from_the_visible_description():
    """'3 cleaning settings/brushing modes ...: super-sensitive for sensitive
    teeth, gentle for gum care, and daily clean for ...' (lower-case names)."""
    html, soup, ld = _load("oralb", "oralb_io-series-2-clean-protect-rechargeable-electric-toothbrush-starter-kit.html")
    det = oralb._extract_details(html, soup, ld, None, [])
    assert det.mode_count == 3
    assert [m.name for m in det.modes] == ["Super-Sensitive", "Gentle", "Daily Clean"]
    assert det.modes[1].description == "for gum care"


def test_oralb_box_list_is_split_into_items_dimensions_and_weight():
    html, soup, ld = _load("oralb", "oralb_io-series-2-clean-protect-rechargeable-electric-toothbrush-starter-kit.html")
    det = oralb._extract_details(html, soup, ld, None, [])
    assert det.box_items == [
        "Oral-B iO2 Electric Toothbrush",
        "Gum & Sensitive Replacement Brush Head",
        "Ultimate Clean Replacement Brush Head",
        "Charger",
    ]
    assert det.dimensions.startswith("Height:") and det.weight.startswith("Weight:")


def test_split_box_item_strips_notes_and_counts_stay():
    item, dims, weight = d.split_box_item(
        "Gum & Sensitive Replacement Brush Head (We recommend starting with this brush head)"
    )
    assert item == "Gum & Sensitive Replacement Brush Head" and dims is None and weight is None
    assert d.split_box_item("(2) Oral-B iO Ultimate Clean Replacement Brush Heads")[0].startswith("(2) ")


def test_footnotes_standalone_and_inline():
    found = d.extract_footnotes(
        ["Remove plaque* and more *vs a manual brush.", "*vs. a regular manual toothbrush"], "Product description"
    )
    assert [(f["marker"], f["text"]) for f in found] == [("*", "vs a manual brush."), ("*", "vs. a regular manual toothbrush")]


# --- Step 2 helpers added for the Feedback.txt fixes ------------------------------

def test_claim_sentence_not_split_at_vs_abbreviation():
    from normalization.utils import split_sentences

    text = ("The iO Series 2 removes 100% more plaque even in hard-to-reach areas vs. a regular manual "
            "toothbrush with Oral-B’s iO technology: tough on plaque, gentle on gums. Another sentence.")
    first = split_sentences(text)[0]
    assert first.endswith("tough on plaque, gentle on gums.")


def test_disclaimers_link_by_exact_marker_run():
    from normalization.disclaimers import link_disclaimers

    out = link_disclaimers(
        [{"marker": "*", "text": "vs. a manual toothbrush", "source": "product"},
         {"marker": "**", "text": "in 6 weeks", "source": "product"},
         {"marker": "***", "text": "unused", "source": "product"}],
        [{"attribute": "claims", "text": "20x more plaque*"}, {"attribute": "claims", "text": "15x healthier gums**"}],
    )
    assert out["per_statement"] == {"20x more plaque*": ["vs. a manual toothbrush"], "15x healthier gums**": ["in 6 weeks"]}
    assert out["orphan_disclaimers"] == ["unused"]


def test_csv_is_written_with_bom_so_excel_reads_utf8(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_table

    path = tmp_path / "t.csv"
    build_table.write_csv([], path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
