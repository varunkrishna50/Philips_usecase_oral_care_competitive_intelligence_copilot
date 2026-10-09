"""Quality gate over the *written* outputs (CSV tables, evidence.json, saved
text snapshots, manifest). Reading the files - not the in-memory objects -
means the gate checks exactly what a reviewer or downstream step will see.

Errors fail the build; warnings are reported only.
"""
from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .aliases import IDENTICAL_RATIO
from .detail_attributes import _Page

FLAG_COLUMNS = [
    "pressure_sensing", "app_connectivity", "bluetooth_connectivity", "timer",
    "coverage_tracking", "replacement_reminder", "travel_case_included",
]
TRI_STATE = {"true", "false", "not_stated"}
US_HOSTS = {"www.usa.philips.com", "oralb.com", "www.oralb.com"}
MIN_PRODUCTS_PER_BRAND = 8
COVERAGE_WARN_BELOW = 0.15

TEXT_COLUMNS = [
    "description", "timer_description", "pressure_sensing_details", "app_connectivity_details", "bluetooth_details",
    "battery_details", "charging_details", "coverage_tracking_details", "claims", "specifications", "cleaning_technology",
    "brush_head_design", "battery_life_note", "feel_and_noise", "price_positioning", "plaque_removal_claim",
    "gum_health_claim", "included_items", "brush_head_compatibility", "display_type",
]
BOILERPLATE_RX = re.compile(
    r"valid email address|news and special offers|by registering,? you confirm|consumer health data|"
    r"sign[- ]?up (?:&|and) save|subscribe to our newsletter|accept (?:all )?cookies|save up to \$\d+ on superior",
    re.I,
)

# keyword -> (column that should then be filled, rows to look in)
KEYWORD_CHECKS = {
    "cleaning_technology": re.compile(r"micro-?vibrations?|oscillat\w+|sonic technology|sonicare technology|(?:bristle|brush) movements", re.I),
    "intensity_count": re.compile(r"\b\d+\s+intensity\s+(?:settings?|levels?)", re.I),
    "timer": re.compile(r"\btimer\b|pacer|\bbuzz\w*\s+every\s+\d+\s+seconds", re.I),
    "warranty_years": re.compile(r"\b\d+[ -]year (?:limited )?warranty|warranty[^.\n]{0,60}\b\d+ years?", re.I),
    "brush_movements_per_min": re.compile(r"\b\d{1,3}(?:,\d{3})+\s*(?:micro-?vibrations?|(?:brush|bristle) movements)", re.I),
    "bluetooth_connectivity": re.compile(r"bluetooth", re.I),
    "app_connectivity": re.compile(r"\bapp\b", re.I),
    "replacement_reminder": re.compile(r"signal when it.s time to replace|replace (?:refill|brush) head on time|refill change alert|replacement reminder", re.I),
    "coverage_tracking": re.compile(r"where you brush|3d teeth tracking|position detection", re.I),
}


def _fold(text: str) -> str:
    """Matching-only folding (non-breaking hyphens/dashes, curly quotes, nbsp)."""
    return text.translate({0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-", 0x2212: "-",
                           0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x00A0: " "})


def _norm_quotes(text: str) -> str:
    """Comparison-only normalization (curly quotes, dashes, whitespace)."""
    text = text.translate({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})
    text = re.sub("[‐-―‑−]", "-", text)
    return re.sub(r"\s+", " ", text).lower()


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_text(path: Path, cache: dict) -> str | None:
    key = str(path)
    if key not in cache:
        cache[key] = path.read_text(encoding="utf-8", newline="") if path.exists() else None
    return cache[key]


def check_files(processed_dir: Path, raw_dir: Path, root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    wide = _read_csv(processed_dir / "comparison_table.csv")
    ids = [r["product_id"] for r in wide]
    by_id = {r["product_id"]: r for r in wide}
    evidence = {e["id"]: e for e in json.loads((processed_dir / "evidence.json").read_text(encoding="utf-8"))}
    cache: dict = {}

    # -- identity -------------------------------------------------------------------------------
    for pid, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate product_id: {pid}")
    brands = Counter(r["brand"] for r in wide)
    for brand in ("Philips", "Oral-B"):
        if brands.get(brand, 0) < MIN_PRODUCTS_PER_BRAND:
            errors.append(f"only {brands.get(brand, 0)} {brand} products (minimum {MIN_PRODUCTS_PER_BRAND})")
    for r in wide:
        u = urlparse(r["product_url"])
        if u.netloc not in US_HOSTS:
            errors.append(f"{r['product_id']}: non-US/unknown host {u.netloc!r}")
        elif "oralb" in u.netloc and not (u.path.startswith("/en-us/") or u.path.startswith("/products/")):
            errors.append(f"{r['product_id']}: Oral-B URL is not under /en-us/: {r['product_url']}")
        if r.get("product_type", "single") != "single":
            errors.append(f"{r['product_id']}: product_type={r['product_type']!r} must not be in the comparison table")

    # -- long tables ---------------------------------------------------------------------------------
    long_tables = {}
    for name in ("claims_long", "specs_long", "feature_headlines", "included_items_long"):
        path = processed_dir / f"{name}.csv"
        if not path.exists():
            errors.append(f"missing output file {path.name}")
            continue
        long_tables[name] = _read_csv(path)
        for row in long_tables[name]:
            if row["product_id"] not in by_id:
                errors.append(f"{name}: product_id {row['product_id']!r} is not in the wide table")

    # -- tri-state flags, mode_count ------------------------------------------------------------------
    for r in wide:
        for col in FLAG_COLUMNS:
            if r.get(col, "") not in TRI_STATE:
                errors.append(f"{r['product_id']}.{col}: {r.get(col)!r} is not true/false/not_stated")
        modes = json.loads(r["modes"]) if r.get("modes") else []
        if modes:
            if not r.get("mode_count", "").strip():
                errors.append(f"{r['product_id']}: modes known ({len(modes)}) but mode_count is blank")
            elif int(float(r["mode_count"])) != len(modes):
                errors.append(f"{r['product_id']}: mode_count={r['mode_count']} != len(modes)={len(modes)}")

    # -- boilerplate -------------------------------------------------------------------------------------------
    for r in wide:
        for col in TEXT_COLUMNS:
            m = BOILERPLATE_RX.search(r.get(col, ""))
            if m:
                errors.append(f"{r['product_id']}.{col}: widget/consent text leaked: ...{m.group(0)}...")

    # -- out of stock vs price ----------------------------------------------------------------------------------
    for r in wide:
        if r.get("availability", "").lower().startswith("out of stock"):
            for col in ("price", "list_price", "current_price", "original_price", "price_after_discount"):
                if r.get(col, "").strip():
                    errors.append(f"{r['product_id']}: {col}={r[col]} shown while out of stock")

    # -- content-identical rows need an alias_group --------------------------------------------------------------
    attr_cols = [c for c in wide[0].keys() if c in FINGERPRINT_COLUMNS]
    for i, a in enumerate(wide):
        for b in wide[i + 1:]:
            if a["brand"] != b["brand"]:
                continue
            same = sum(1 for c in attr_cols if a.get(c) == b.get(c))
            if attr_cols and same / len(attr_cols) >= IDENTICAL_RATIO:
                if not a.get("alias_group") or a.get("alias_group") != b.get("alias_group"):
                    errors.append(f"{a['product_id']} and {b['product_id']} are content-identical ({same}/{len(attr_cols)}) but share no alias_group")

    # -- quotes found at their offsets ---------------------------------------------------------------------------------
    def check_quote(eid: str, label: str, severity_list: list[str]) -> None:
        ev = evidence.get(eid)
        if ev is None:
            severity_list.append(f"{label}: evidence id {eid!r} not in evidence.json")
            return
        if ev.get("source_offset") is None or not ev.get("snapshot_file"):
            severity_list.append(f"{label}: {eid} has no snapshot offset")
            return
        snap = Path(ev["snapshot_file"])
        text = _read_text(snap if snap.is_absolute() else root / snap, cache)
        if text is None:
            severity_list.append(f"{label}: snapshot file missing for {eid}: {ev['snapshot_file']}")
            return
        off, quote = ev["source_offset"], ev["evidence_text"]
        if text[off: off + len(quote)] != quote:
            severity_list.append(f"{label}: quote not found at offset {off} in {ev['snapshot_file']} ({quote[:60]!r})")

    for row in long_tables.get("claims_long", []):
        if not row.get("source_section"):
            errors.append(f"claims_long {row['product_id']}: claim without source_section: {row['claim_text'][:60]!r}")
        if not row.get("evidence_id"):
            errors.append(f"claims_long {row['product_id']}: claim without evidence id: {row['claim_text'][:60]!r}")
        else:
            check_quote(row["evidence_id"], f"claims_long {row['product_id']}", errors)
        if row.get("figure") and not row.get("comparator"):
            warnings.append(f"{row['product_id']}: claim with a figure but no comparator: {row['claim_text'][:70]!r}")
    # -- feature headlines: clean text, found at their offset -----------------------------------------------------
    for row in long_tables.get("feature_headlines", []):
        h = row["headline"]
        where = f"feature_headlines {row['product_id']}: {h[:50]!r}"
        if "\\n" in h or "\n" in h or h.endswith(("-", "–", "—", ":")) or len(h.split()) < 2:
            errors.append(f"{where}: headline still has an extraction artefact")
        if not row.get("snapshot_file") or row.get("source_offset", "") == "":
            errors.append(f"{where}: no snapshot offset")
            continue
        snap = Path(row["snapshot_file"])
        text = _read_text(snap if snap.is_absolute() else root / snap, cache)
        off = int(row["source_offset"])
        quote = h.replace("&", "\\u0026") if row.get("quote_form") == "json_escaped" else h
        if text is None or text[off: off + len(quote)] != quote:
            errors.append(f"{where}: headline not found at offset {off} in {row['snapshot_file']}")

    # -- Philips drawer: specs_long holds exactly the drawer's sections -------------------------------------------
    for pid in by_id:
        if not pid.startswith("philips_"):
            continue
        record_path = raw_dir / "philips" / f"{pid}.json"
        if not record_path.exists():
            continue
        page = _Page(json.loads(record_path.read_text(encoding="utf-8")))
        expected = {
            html.unescape(cname) for cname, (obj, _r) in page.chapters.items()
            if any(item.get("csValue") and any(v.get("csValueName") for v in item["csValue"]) for item in obj.get("csItem", []))
        }
        got = {r["group"] for r in long_tables.get("specs_long", [])
               if r["product_id"] == pid and r.get("origin") == "drawer" and r.get("source_section", "").startswith("spec:")}
        if expected != got:
            errors.append(f"{pid}: specs_long drawer sections differ from the page: missing {sorted(expected - got)}, extra {sorted(got - expected)}")

    # -- brush head count agrees in the wide table, specs_long and included_items_long -------------------------
    for pid, r in by_id.items():
        wide_n = r.get("brush_heads_included", "").strip()
        if not wide_n:
            continue
        spec_rows = [x for x in long_tables.get("specs_long", []) if x["product_id"] == pid and x["name"] == "Brush heads included"]
        item_n = sum(int(x["quantity"]) if str(x["quantity"]).strip() else 1
                     for x in long_tables.get("included_items_long", []) if x["product_id"] == pid and x["kind"] == "brush_head")
        values = {int(float(wide_n)), item_n} | {int(float(x["value"])) for x in spec_rows}
        if len(values) > 1:
            errors.append(f"{pid}: brush head count differs between tables: wide={wide_n}, included_items_long={item_n}, "
                          f"specs_long={[x['value'] for x in spec_rows]}")

    # -- long-table keys, units, quantities -------------------------------------------------------------------------
    from .longtables import UNIT_WHITELIST

    for name, table in long_tables.items():
        ids_seen = Counter(r.get("row_id", "") for r in table)
        for rid, n in ids_seen.items():
            if not rid:
                errors.append(f"{name}: a row has no row_id")
            elif n > 1:
                errors.append(f"{name}: row_id {rid!r} is used {n} times")
    for r in long_tables.get("specs_long", []):
        if r.get("unit", "") not in UNIT_WHITELIST:
            errors.append(f"specs_long {r['product_id']}: unit {r['unit']!r} is not a unit (row {r['name']!r})")
        if r.get("item_text") and r.get("group") != "Items included":
            errors.append(f"specs_long {r['product_id']}: item_text outside the Items included section ({r['name']!r})")
    repeats = Counter((r["product_id"], r["group"], r["name"], r["value_raw"]) for r in long_tables.get("specs_long", []))
    for (pid, group, name, raw), n in repeats.items():
        if n > 1:
            warnings.append(f"{pid}: specs_long repeats {group} / {name} = {raw!r} {n} times")
    for r in long_tables.get("included_items_long", []):
        if r.get("quantity_source") not in ("stated", "assumed_default"):
            errors.append(f"included_items_long {r['product_id']}: quantity_source {r.get('quantity_source')!r} is not stated/assumed_default")
        elif r["quantity_source"] == "stated" and not str(r.get("quantity", "")).strip():
            errors.append(f"included_items_long {r['product_id']}: stated quantity is blank for {r['item']!r}")
    for r in wide:
        if r.get("availability") in ("In stock", "Out of stock"):
            if not r.get("availability_raw"):
                errors.append(f"{r['product_id']}: availability {r['availability']!r} has no availability_raw (the site's own wording)")
            av_ev = [e for e in evidence.values() if e["product_id"] == r["product_id"] and e["attribute"] == "availability"]
            if av_ev and not all(e.get("evidence_tier") == "cited" for e in av_ev) and \
                    (processed_dir.parent / "raw" / "text" / r["brand"].lower().replace("-", "") / f"{r['product_id']}.rendered.txt").exists():
                errors.append(f"{r['product_id']}: availability is cited from page data, not from the visible page")

    for eid, ev in evidence.items():
        if ev.get("source_offset") is not None:
            check_quote(eid, "evidence", errors)
        else:
            warnings.append(f"evidence {eid}: no snapshot offset")
    for eid, ev in evidence.items():
        if not ev.get("source_section") and not ev.get("region"):
            errors.append(f"evidence {eid}: no source_section/region")

    # -- warnings ------------------------------------------------------------------------------------------------------------
    for r in wide:
        if r.get("app_connectivity") == "true" and r.get("bluetooth_connectivity") == "not_stated":
            warnings.append(f"{r['product_id']}: app_connectivity=true but bluetooth_connectivity=not_stated")
    for col in wide[0].keys():
        filled = sum(1 for r in wide if r[col].strip() and r[col] != "not_stated")
        if filled / len(wide) < COVERAGE_WARN_BELOW:
            warnings.append(f"column {col}: filled for {filled}/{len(wide)} products (< {int(COVERAGE_WARN_BELOW*100)}%)")

    for r in wide:
        pid = r["product_id"]
        pieces = []
        for rel in (r.get("text_snapshot"),):
            if rel:
                t = _read_text(root / rel, cache)
                if t:
                    # comparison FAQs mix products, menus/review widgets are not product copy
                    skip = r"(?:difference|\bvs\b|versus|sort by|page content|reviews?|the perfect brush)"
                    pieces.append(re.sub(r"=== \[\d+\] [^\n]*" + skip + r"[^\n]* ===\n.*?(?=\n=== \[|\Z)", "", t, flags=re.S | re.I))
        structured = _read_text(processed_dir / "snapshots" / f"{pid}.structured.txt", cache)
        if structured:
            for block in re.split(r"(?m)^### ", structured):
                header = block.split("\n", 1)[0]
                label = header.split(" [", 1)[0]
                if ("[visible]" in header or "[expanded]" in header) and (label.startswith(("spec:", "feature_panels", "rendered"))):
                    pieces.append(block)
        visible = _fold("\n".join(pieces))
        for col, rx in KEYWORD_CHECKS.items():
            val = r.get(col, "")
            blank = (not val.strip()) or val == "not_stated"
            if blank and rx.search(visible):
                m = rx.search(visible)
                warnings.append(f"{pid}.{col} is blank but the visible text says: ...{visible[max(0, m.start()-40):m.end()+40].strip()!r}...")
        for key, val in json.loads(r.get("variant_values") or "{}").items():
            warnings.append(f"{pid}: variant pages disagree on {key}: {val}")

    cross = sum(1 for e in evidence.values() if e.get("evidence_tier") == "cross_check_only")
    if cross:
        warnings.append(f"{cross} evidence rows are cross_check_only (structured data, kept but not cited)")
    return errors, warnings


# Columns that carry page-derived attribute values (identity/provenance excluded).
FINGERPRINT_COLUMNS = {
    "price", "currency", "mode_count", "modes", "pressure_sensing", "app_connectivity", "bluetooth_connectivity",
    "timer", "timer_description", "battery_life_days", "included_items", "description", "claims", "specifications",
    "colors", "mode_details", "warranty_years", "battery_type", "plaque_removal_claim", "gum_health_claim",
    "cleaning_technology", "pressure_sensing_details", "battery_details", "dimensions", "weight",
}
