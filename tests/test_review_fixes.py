"""Regression tests for the data-quality review fixes, run against the built outputs
(data/processed/*.csv and the saved text snapshots)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


def norm(s: str) -> str:
    """Compare-only normalisation: quotes, dashes, whitespace, case."""
    s = re.sub(r"[‘’ʼ]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‐-―−]", "-", s)
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip().lower()


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(PROC / name, encoding="utf-8-sig", dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return read("comparison_table.csv").set_index("product_id", drop=False)


def pid(table, part):
    hits = [p for p in table.index if part in p]
    assert len(hits) == 1, (part, hits)
    return hits[0]


OB = "oralb_io-series-"


def snapshot(product_id: str) -> str:
    brand = product_id.split("_")[0]
    return (RAW / "text" / brand / f"{product_id}.txt").read_text(encoding="utf-8", newline="")


def test_io4_faq_sentence_survives_cleaning():
    text = norm(snapshot(OB + "4-rechargeable-electric-toothbrush").join(["", ""]))
    assert "combines oscillating & rotating movements with gentle micro-vibrations" in text


@pytest.mark.parametrize("part", ["series-2-", "series-6-", "series-10-"])
def test_oralb_cleaning_technology_is_verbatim(table, part):
    p = pid(table, part)
    sentence = table.loc[p, "cleaning_technology"]
    assert sentence and norm(sentence) in norm(snapshot(p))


def test_io9_cleaning_technology_and_head_design(table):
    p = OB + "9-rechargeable-electric-toothbrush"
    assert "micro-vibrations" in norm(table.loc[p, "cleaning_technology"])
    assert table.loc[p, "brush_head_design"]


def test_philips_3100_intensity_count(table):
    assert table.loc[pid(table, "hx3681"), "intensity_count"] in ("2", "2.0")


def test_replacement_reminder_from_device_statements(table):
    cgc = pid(table, "series-4-clean-gum-care")
    io6 = pid(table, "series-6-")
    assert table.loc[cgc, "replacement_reminder"] == "true"
    assert table.loc[io6, "replacement_reminder"] == "true"
    assert "signal when it's time to replace your brush head" in norm(snapshot(cgc))
    assert "replace refill head on time, and track battery level" in norm(snapshot(io6))


def test_replacement_advice_alone_does_not_set_reminder(table):
    io3 = pid(table, "series-3-")
    assert table.loc[io3, "replacement_reminder"] != "true" or re.search(
        r"signal|alert|remind|track|indicator|monitor", norm(table.loc[io3, "replacement_reminder_details"]
        if "replacement_reminder_details" in table else snapshot(io3)))


@pytest.mark.parametrize("part", ["series-5-", "series-6-", "series-8-"])
def test_coverage_tracking(table, part):
    assert table.loc[pid(table, part), "coverage_tracking"] == "true"


def test_philips_flags_from_spec_rows(table):
    assert table.loc[pid(table, "hx7410"), "timer"] == "true"
    assert table.loc[pid(table, "hx9990"), "bluetooth_connectivity"] == "true"


def test_no_widget_text_in_text_columns(table):
    pattern = re.compile(r"sign up for|subscribe to our newsletter|accept all cookies|cookie settings", re.I)
    for col in table.columns:
        for v in table[col]:
            assert not pattern.search(v), (col, v[:80])


def test_hx3681_mode_count(table):
    assert table.loc[pid(table, "hx3681"), "mode_count"] in ("1", "1.0")


def test_twin_pack_excluded_and_single_kept(table):
    assert not any(re.search(r"twin|dual|duo|bundle", p) for p in table.index)
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    assert "twin" in json.dumps(manifest).lower() or "bundle" in json.dumps(manifest).lower()
    assert pid(table, "series-2-")  # 1 handle / 2 heads stays


def test_hx4031_claims_have_5x_and_700(table):
    claims = read("claims_long.csv")
    rows = claims[claims.product_id == pid(table, "hx4031")]
    figs = " ".join(rows.claim_text + " " + rows.figure)
    assert "5x" in figs.lower() and "700%" in figs


def test_mode_canonical_equivalence():
    from normalization.vocab import canonical_mode, compare_mode_names
    assert compare_mode_names("Whiten", "Whitening") == "similar"
    assert compare_mode_names("Whitening", "White+") == "similar"
    assert compare_mode_names("Super-Sensitive", "Super Sensitive") == "same"
    assert compare_mode_names("Sensitive", "Super Sensitive") == "different"
    assert canonical_mode("Whitening")["equivalence"] == "approximate"


def test_genius_aliases_to_io9(table):
    g = OB + "9-genius-rechargeable-electric-toothbrush"
    assert table.loc[g, "canonical_product_id"] == OB + "9-rechargeable-electric-toothbrush"
    assert table.loc[g, "alias_group"]


def test_product_aliases_resolve():
    al = read("product_aliases.csv")
    for q, expect in [("hx3681", "philips_hx3681-03"), ("3100 series", "philips_hx3681-03"),
                      ("io6", OB + "6-electric-toothbrush")]:
        hits = al[al.normalized_alias == q]
        assert not hits.empty and expect in set(hits.canonical_product_id), q


@pytest.mark.parametrize("name", ["claims_long.csv", "specs_long.csv", "included_items_long.csv"])
def test_offsets_match_snapshots(name):
    evidence = {e["id"]: e for e in json.loads((PROC / "evidence.json").read_text(encoding="utf-8"))}
    df = read(name)
    df = df[df.source_offset != ""]
    assert len(df) > 0
    for _, r in df.iterrows():
        assert not re.match(r"^[A-Za-z]:", r.snapshot_file), r.snapshot_file  # relative paths only
        text = (ROOT / r.snapshot_file).read_text(encoding="utf-8", newline="")
        quote = evidence[r.evidence_id]["evidence_text"]
        off = int(r.source_offset)
        assert text[off:off + len(quote)] == quote, (name, r.evidence_id)


def test_claims_have_real_section_names():
    claims = read("claims_long.csv")
    assert not claims.source_section.isin(["rendered_text", "visible_text", "structured_data", ""]).any()


def test_philips_one_timer_movements_warranty_and_claims(table):
    one = pid(table, "hy1200")
    assert table.loc[one, "timer"] == "true"
    assert table.loc[one, "brush_movements_per_min"] in ("13000", "13000.0")
    assert table.loc[one, "warranty_years"] in ("2", "2.0")
    claims = read("claims_long.csv")
    for part in ("hy1200", "hy1130"):
        assert (claims.product_id == pid(table, part)).sum() >= 2


def test_9900_gum_claim_uses_six_weeks_not_two_days(table):
    claims = read("claims_long.csv")
    rows = claims[(claims.product_id == pid(table, "hx9990")) & (claims.topic == "gums")]
    assert not rows.empty
    assert not rows.claim_text.str.contains(r"\*\*\*").any()
    assert rows.timeframe.str.contains("6 weeks|six weeks").any()
    assert not rows.timeframe.str.contains("2 days|two days").any()


def test_multi_figure_claims_get_one_row_per_figure(table):
    claims = read("claims_long.csv")
    for part, plaque, gums in (("hx7130", "1000%", "100%"), ("hx7421", "1000%", "600%")):
        rows = claims[(claims.product_id == pid(table, part)) & claims.claim_text.str.startswith("Our Next-Generation")]
        by_topic = dict(zip(rows.topic, rows.figure.str.replace("up to ", "", case=False)))
        assert by_topic == {"plaque": plaque, "gums": gums}, (part, by_topic)


def test_comparator_is_never_a_time_span():
    claims = read("claims_long.csv")
    assert not claims.comparator.str.contains(r"than (?:two|\d)", case=False, regex=True).any()


def test_contradictory_claims_are_flagged_not_dropped(table):
    claims = read("claims_long.csv")
    for part, figs in (("hx4031", {"700%", "5x"}), ("series-3-", {"6x", "400%"})):
        rows = claims[(claims.product_id == pid(table, part)) & (claims.conflict != "")]
        assert figs <= {f.lower().replace("up to ", "") for f in rows.figure}, part


def test_intensity_count_matches_levels(table):
    import json as _json
    for p in table.index:
        levels = table.loc[p, "intensity_levels"]
        if levels.startswith("["):
            assert float(table.loc[p, "intensity_count"]) == len(_json.loads(levels)), p


def test_brush_head_design_is_a_short_phrase(table):
    for p in table.index:
        v = table.loc[p, "brush_head_design"]
        assert len(v) <= 60, (p, v)


def test_iO10_charge_time_not_in_battery_note(table):
    p = pid(table, "series-10-")
    assert "takes about 3 hours" not in table.loc[p, "battery_life_note"]
    assert table.loc[p, "charge_time_hours"] in ("3", "3.0")


def test_philips_replacement_reminder_needs_a_statement(table):
    ev = {e["id"]: e for e in json.loads((PROC / "evidence.json").read_text(encoding="utf-8"))}
    for p in table.index:
        if p.startswith("philips") and table.loc[p, "replacement_reminder"] == "true":
            e = [x for x in ev.values() if x["product_id"] == p and x["attribute"] == "replacement_reminder"]
            assert e and "replace" in e[0]["evidence_text"].lower(), p


def test_iO8_charge_time_and_io5_brush_head_tracker(table):
    assert table.loc[pid(table, "series-8-"), "charge_time_hours"] in ("3", "3.0")
    assert table.loc[pid(table, "series-5-"), "replacement_reminder"] != "true"


def test_aliases_flag_ambiguous_io4_and_cover_philips_families():
    al = read("product_aliases.csv")
    assert (al[al.normalized_alias == "io4"].ambiguous == "true").all()
    for q in ("io4 clean gum care", "9300 series", "sonicare 9000", "prestige", "9900 series", "philips one", "one for kids"):
        hits = al[al.normalized_alias == q]
        assert not hits.empty and (hits.ambiguous != "true").all(), q


def test_flags_are_tristate(table):
    for col in ["pressure_sensing", "app_connectivity", "bluetooth_connectivity", "timer",
                "replacement_reminder", "coverage_tracking"]:
        assert set(table[col]) <= {"true", "false", "not_stated"}, col


# --- long-table fixes (warranty, drawer sections, brush heads, kinds, headlines, aliases) ---------------------

def test_split_value_unit_reads_n_year():
    from normalization.longtables import split_value_unit
    assert split_value_unit("2-year limited warranty") == ("2", "years")
    assert split_value_unit("2 year warranty") == ("2", "years")
    assert split_value_unit("100-240") == ("100-240", "")
    assert split_value_unit("21 days") == ("21", "days")


def test_one_warranty_and_one_battery_type_row_per_philips_product():
    specs = read("specs_long.csv")
    ph = specs[specs.product_id.str.startswith("philips")]
    for name in ("Warranty", "Battery type"):
        counts = ph[ph.name == name].groupby("product_id").size()
        assert not counts.empty and counts.max() == 1, name
    w = ph[ph.name == "Warranty"]
    assert set(w.value) == {"2"} and set(w.unit) == {"years"}
    assert w.value_raw.str.contains("2.year", regex=True).all()
    # the drawer's own wording is kept (no merged "Lithium-ion" duplicate)
    assert "Lithium ION" in set(ph[ph.name == "Battery type"].value_raw)
    assert "Lithium-ion" not in set(ph[ph.name == "Battery type"].value_raw)


def test_drawer_sections_are_all_in_specs_long():
    import html
    from normalization.detail_attributes import _Page
    specs = read("specs_long.csv")
    for path in sorted((RAW / "philips").glob("philips_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        page = _Page(record)
        expected = {html.unescape(c) for c, (obj, _r) in page.chapters.items()
                    if any(v.get("csValueName") for it in obj.get("csItem", []) for v in it.get("csValue", []))}
        rows = specs[(specs.product_id == path.stem) & (specs.origin == "drawer") & specs.source_section.str.startswith("spec:")]
        assert set(rows.group) == expected, path.stem
    rows = specs[(specs.product_id == "philips_hx7130-01") & (specs.group == "Smart sensor technology")]
    assert {"Pressure feedback", "Replacement Reminder"} <= set(rows.name)
    assert rows.source_section.str.startswith("spec:").all()


def test_hx7130_mode_count_is_cited_to_the_modes_drawer():
    ev = [e for e in json.loads((PROC / "evidence.json").read_text(encoding="utf-8"))
          if e["product_id"] == "philips_hx7130-01" and e["attribute"] == "mode_count"]
    assert ev and ev[0]["region"] == "spec:Modes"
    assert "healthier gums" not in ev[0]["evidence_text"]


def test_brush_head_helper_splits_semicolon_lines():
    from normalization.included import brush_head_total, parse_included
    line = "Brush head: 1 C3 Premium Plaque Control; 1 S2 Sensitive"
    assert [(e["item"], e["quantity"], e["kind"]) for e in parse_included(line)] == [
        ("C3 Premium Plaque Control", 1, "brush_head"), ("S2 Sensitive", 1, "brush_head")]
    assert brush_head_total([line]) == 2
    assert brush_head_total(["(2) Brush Heads"]) == 2
    assert brush_head_total(["Gum & Sensitive Replacement Brush Head", "Ultimate Clean Replacement Brush Head"]) == 2
    assert parse_included("Accessory: 1 Toothbrush stand")[0]["kind"] == "accessory"
    assert parse_included("Toothbrush stand")[0]["kind"] == "accessory"


def test_brush_head_count_agrees_in_all_three_places(table):
    specs, items = read("specs_long.csv"), read("included_items_long.csv")
    assert table.loc[pid(table, "hx7410"), "brush_heads_included"] in ("2", "2.0")
    assert table.loc[pid(table, "hx9903"), "brush_heads_included"] in ("2", "2.0")
    for p in table.index:
        wide = table.loc[p, "brush_heads_included"]
        if not wide:
            continue
        sheet = specs[(specs.product_id == p) & (specs.name == "Brush heads included")]
        n = sum(int(q) if q else 1 for q in items[(items.product_id == p) & (items.kind == "brush_head")].quantity)
        assert {int(float(wide)), n, *[int(float(v)) for v in sheet.value]} == {n}, p


def test_item_kinds_follow_the_row_label():
    items = read("included_items_long.csv")
    stand = items[(items.product_id == "philips_hy1130-02") & (items.item == "Toothbrush stand")]
    assert list(stand.kind) == ["accessory"]
    assert not (items.kind == "handle")[items.item.str.contains("stand", case=False)].any()


def test_feature_headlines_are_clean_and_traceable():
    f = read("feature_headlines.csv")
    assert not f.headline.str.contains("\\n", regex=False).any()  # no literal backslash-n
    assert not f.headline.str.contains("\n", regex=False).any()
    assert not f.headline.str.contains("[-–—:\\\\]$", regex=True).any()
    assert (f.headline.str.split().str.len() >= 2).all()
    for col in ("evidence_id", "snapshot_file", "source_offset", "visibility"):
        assert (f[col] != "").all(), col
    for _, r in f.iterrows():
        text = (ROOT / r.snapshot_file).read_text(encoding="utf-8", newline="")
        quote = r.headline.replace("&", "\\u0026") if r.quote_form == "json_escaped" else r.headline
        assert text[int(r.source_offset):int(r.source_offset) + len(quote)] == quote, r.evidence_id
    # two-word feature names are real features and stay
    assert f.headline.str.lower().isin(["pressure sensor", "replacement reminder"]).any()


def test_csv_flag_cells_are_lowercase_in_the_raw_file():
    import csv
    with (PROC / "comparison_table.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for col in ("timer", "pressure_sensing", "app_connectivity", "bluetooth_connectivity",
                "replacement_reminder", "coverage_tracking"):
        assert {r[col] for r in rows} <= {"true", "false", "not_stated"}, col


def test_3100_alias_is_ambiguous_with_a_slug_note():
    al = read("product_aliases.csv")
    hits = al[al.normalized_alias == "3100 series"]
    assert set(hits.canonical_product_id) == {"philips_hx3681-03", "philips_hx4031-71"}
    assert (hits.ambiguous == "true").all()
    assert hits.candidates.str.contains("NOTE: hx4031-71 is titled").all()
    assert hits.candidates.str.contains("URL slug says '3100 series'").all()


# --- second review: units, keys, quantities, availability, brand filler ---------------------------------------

def test_parse_spec_value_keeps_source_and_normalises():
    from normalization.longtables import UNIT_WHITELIST, parse_spec_value
    assert parse_spec_value("Up to 14 days") == {"value": "14", "unit": "days", "qualifier": "up_to", "item_text": ""}
    assert parse_spec_value("62,000 brush movements / min")["unit"] == "per min"
    assert parse_spec_value("62,000 brush movements / min")["value"] == "62000"
    assert parse_spec_value("2-year limited warranty")["unit"] == "years"
    one = parse_spec_value("1 Prestige USB charging travel case", "Items included")
    assert one == {"value": "1", "unit": "count", "qualifier": "", "item_text": "Prestige USB charging travel case"}
    assert parse_spec_value("1 AAA Alkaline", "Technical specifications")["unit"] == ""  # not a unit
    assert parse_spec_value("Lithium ION")["unit"] in UNIT_WHITELIST


def test_specs_units_are_whitelisted_and_items_move_to_item_text():
    from normalization.longtables import UNIT_WHITELIST
    specs = read("specs_long.csv")
    assert set(specs.unit) <= UNIT_WHITELIST, set(specs.unit) - UNIT_WHITELIST
    one = specs[(specs.product_id == "philips_hy1200-25") & (specs.group == "Items included") & (specs.name == "Handle")]
    assert list(one.value) == ["1"] and list(one.unit) == ["count"] and list(one.item_text) == ["Philips One"]
    assert (specs[specs.group != "Items included"].item_text == "").all()
    assert "brush movements / min" not in set(specs.unit)


def test_up_to_qualifier_and_raw_text_survive():
    specs = read("specs_long.csv")
    for pid_, value, raw in (("philips_hx3681-03", "14", "Up to 14 days"), ("philips_hy1200-25", "30", "Up to 30 days")):
        row = specs[(specs.product_id == pid_) & specs.name.str.startswith("Operating time")]
        assert list(row.value) == [value] and list(row.unit) == ["days"], pid_
        assert list(row.qualifier) == ["up_to"] and list(row.value_raw) == [raw], pid_
    speed = specs[(specs.name == "Speed") & specs.value_raw.str.contains("62,000")]
    assert not speed.empty and set(speed.value) == {"62000"} and set(speed.unit) == {"per min"}
    assert speed.value_raw.str.contains("brush movements / min").all()


def test_row_ids_are_unique_in_all_long_tables():
    for name in ("claims_long", "specs_long", "feature_headlines", "included_items_long"):
        df = read(f"{name}.csv")
        assert (df.row_id != "").all() and df.row_id.is_unique, name
    items = read("included_items_long.csv")
    split = items[items.item_raw.str.contains(";")]
    assert split.evidence_id.duplicated().any() and split.row_id.is_unique  # shared FK, distinct primary keys


def test_hy1130_battery_rows_are_distinct():
    specs = read("specs_long.csv")
    rows = specs[(specs.product_id == "philips_hy1130-02") & (specs.name == "Battery")]
    assert len(rows) == 2 and rows.row_id.is_unique
    assert set(rows.group) == {"Technical specifications", "Items included"}
    box = rows[rows.group == "Items included"].iloc[0]
    assert (box.value, box.unit, box.item_text) == ("1", "count", "AAA Alkaline")
    dup = specs.duplicated(["product_id", "group", "name", "value_raw"], keep=False)
    assert not dup.any()


def test_blank_quantities_are_marked_assumed():
    items = read("included_items_long.csv")
    assert set(items.quantity_source) <= {"stated", "assumed_default"}
    assert (items.quantity != "").all()
    iO2 = items[items.product_id.str.contains("series-2-")]
    assert set(iO2.quantity_source) == {"assumed_default"} and set(iO2.quantity) == {"1"}
    philips = items[items.product_id == "philips_hx7410-10"]
    assert set(philips.quantity_source) == {"stated"}


def test_brand_filler_headlines_are_marked_not_dropped():
    f = read("feature_headlines.csv")
    filler = f[f.kind == "brand_filler"]
    text = " | ".join(filler.headline).lower()
    for phrase in ("trust oral", "make the most of every session", "switch to oral-b electric toothbrushes"):
        assert phrase in text, phrase
    assert not filler.headline.str.lower().str.contains("pressure sensor|timer").any()


def test_availability_is_cited_from_the_visible_buy_box(table):
    ev = {e["product_id"]: e for e in json.loads((PROC / "evidence.json").read_text(encoding="utf-8"))
          if e["attribute"] == "availability"}
    for p in table.index:
        status, raw = table.loc[p, "availability"], table.loc[p, "availability_raw"]
        if p.startswith("oralb"):
            assert status in ("In stock", "Out of stock"), p
            if status == "In stock":
                assert raw.startswith("Add to cart"), (p, raw)
            else:
                assert raw == "Find A Retailer", (p, raw)
            assert ev[p]["evidence_text"] == raw and ev[p]["evidence_tier"] == "cited", p
    assert table.loc[pid(table, "series-10-"), "availability"] == "Out of stock"
    assert table.loc[pid(table, "genius"), "availability_raw"].startswith("Add to cart")
    assert table.loc[pid(table, "genius"), "source_variant_label"]  # the colour the status applies to


def test_availability_without_a_visible_cue_is_not_stated():
    from normalization.detail_extras import _visible_availability

    class FakePage:
        def __init__(self, brand, rendered):
            self.brand, self.rendered = brand, rendered

    assert _visible_availability(FakePage("Oral-B", "intro\n\n=== Buy box ===\nAdd to cart ($79.99)\n")) == ("in_stock", "Add to cart ($79.99)")
    assert _visible_availability(FakePage("Oral-B", "intro\n\n=== Buy box ===\nFind A Retailer\n")) == ("out_of_stock", "Find A Retailer")
    assert _visible_availability(FakePage("Oral-B", "intro\n\n=== Buy box ===\nemail\nColors:\n")) is None
    assert _visible_availability(FakePage("Oral-B", "Find A Retailer in the footer, no buy box block")) is None


def test_app_and_bluetooth_details_exist_wherever_the_flag_is_true(table):
    """The *_details prose is gated on the flag; flags found only in a feature panel / spec row
    (hx7410, hx7421, hx9990) must still get their details."""
    for p in table.index:
        if table.loc[p, "app_connectivity"] == "true":
            assert table.loc[p, "app_connectivity_details"], p
        if p.startswith("philips") and table.loc[p, "bluetooth_connectivity"] == "true":
            assert table.loc[p, "bluetooth_details"], p
    assert "app" in table.loc[pid(table, "hx9990"), "app_connectivity_details"].lower()
