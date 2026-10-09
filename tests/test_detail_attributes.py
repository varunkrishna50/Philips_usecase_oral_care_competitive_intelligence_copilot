"""Step 2 detail attributes (pricing detail, colors, ratings, specs, flags)
built from the real Step 1 records. Every stated value must carry
evidence that is a verbatim substring of its source -- validate() checks
that independently from the raw files."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normalization.detail_attributes import DETAIL_ATTRIBUTES  # noqa: E402
from normalization.models import ATTRIBUTE_FIELDS  # noqa: E402
from normalization.normalize import index_raw_records, load_raw_records, normalize_all  # noqa: E402
from normalization.validation import validate  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def built():
    records = [r for r in load_raw_records(RAW_DIR) if r.get("details")]
    if not records or not Path(records[0]["raw_html_path"]).exists():
        pytest.skip("raw records / HTML snapshots not available (run from repo root)")
    products, store, _ = normalize_all(records)
    return products, store, records


def _by_id(products, pid):
    return next(p for p in products if p.product_id == pid)


def test_detail_attributes_are_registered_fields():
    assert set(DETAIL_ATTRIBUTES) <= set(ATTRIBUTE_FIELDS)


def test_validation_passes_with_detail_evidence(built):
    products, store, records = built
    assert validate(products, store, index_raw_records(records)) == []


def test_philips_prestige_pricing_and_specs(built):
    products, store, _ = built
    p = _by_id(products, "philips_hx9990-13")
    assert p.original_price.value == 399.96
    assert p.price_after_discount.value == 379.96
    assert p.discount_pct.value == 5.0 and p.discount_amount.value == 20.0
    assert p.color_count.value == 3
    assert (p.rating_avg.value, p.rating_count.value) == (4.2, 488)
    assert p.warranty_years.value == 2
    assert p.battery_type.value == "Lithium-ion"
    assert p.brush_movements_per_min.value == 62000
    assert [m["name"] for m in p.mode_details.value][:2] == ["Clean", "Deep Clean"]
    assert p.travel_case_included.value is True
    assert p.intensity_levels.value == ["High", "Medium", "Low"]
    # Evidence is clean JSON (one escape level removed), not backslash-escaped.
    ev = store.get(p.original_price.evidence_ids[0])
    assert '"discountPrice"' in ev.evidence_text and "\\" not in ev.evidence_text


def test_oralb_io9_pricing_and_flags(built):
    products, store, _ = built
    p = _by_id(products, "oralb_io-series-9-rechargeable-electric-toothbrush")
    assert p.original_price.value == 329.99 and p.price.value == 329.99  # regular price is the primary price
    assert p.price_after_discount.value <= p.original_price.value
    assert p.mode_details.value[0]["name"] == "Daily Clean"
    assert p.warranty_years.value == 2
    assert p.charge_time_hours.value == 3.0
    assert p.coverage_tracking.value is True
    assert p.display_type.value.lower().startswith("interactive")
    assert p.rating_avg.value is not None and p.rating_count.value is not None
    ev = store.get(p.rating_avg.evidence_ids[0])
    assert "Reviews" in ev.evidence_text


def test_missing_facts_stay_not_mentioned(built):
    products, _, _ = built
    p = _by_id(products, "philips_hy1200-25")
    assert p.warranty_years.status == "not_mentioned" and p.warranty_years.value is None
    assert p.coverage_tracking.status == "not_mentioned"


def test_no_stated_value_without_evidence(built):
    products, _, _ = built
    for p in products:
        for name in DETAIL_ATTRIBUTES:
            av = getattr(p, name)
            if av.status == "stated":
                assert av.value is not None and av.evidence_ids, f"{p.product_id}.{name}"
            else:
                assert av.value is None and not av.evidence_ids
