"""Long-format companion tables to the wide comparison table. Every row
carries ``product_id`` (a key into the wide table) and, where the value is
evidence-backed, the evidence id plus the snapshot location of the quote.

  claims_long            one row per distinct claim (never collapsed)
  specs_long             group / name / value / unit
  feature_headlines      marketing headlines (feature panels, bold lead-ins)
  included_items_long    box contents with quantity and kind
"""
from __future__ import annotations

import html
import re
from typing import Any, Optional

from .disclaimers import marker_runs
from .evidence import EvidenceStore
from .models import AttributeValue, NormalizedProduct

_FIGURE_RX = re.compile(r"(?P<upto>up to\s+)?(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>x|%)", re.I)
_NUMWORD = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)"
# a comparator is what the figure is measured against; a time span after "than" ("than two days") is not one
_COMPARATOR_RX = re.compile(
    r"(?:\bvs\.?|\bversus|compared (?:to|with)|(?<!less )\bthan)\s+(?!" + _NUMWORD + r"\b)(?:a |an |the )?"
    r"[A-Za-z\- ]{3,40}?(?=[,.;:*)]|\s+(?:in|with|and|for|within|using)\b|$)", re.I
)
_TIMEFRAME_RX = re.compile(
    r"\b(?:in|within|after)\s+(?:just |only |as little as )?(?:less than |under )?" + _NUMWORD
    + r"\s*(?:weeks?|days?|months?|minutes?|seconds?)\b|\bfrom day one\b", re.I
)
_CONDITION_RX = re.compile(
    r"\bin [A-Za-z+\- ]{2,30}? mode\b|\busing [^.;]{3,50}?(?=\s+in\s+\w+\s+(?:days?|weeks?)|[.;]|$)|\bin hard-to-reach areas\b", re.I
)
_TOPICS = (("plaque", r"plaque"), ("gums", r"\bgum"), ("stains/whitening", r"stain|whiten|white"))


def assign_row_ids(rows: list[dict], table: str) -> None:
    """Primary key of a long-table row: <product_id>:<table>:<n>, n counted per product.
    ``evidence_id`` stays the foreign key into evidence.json (several rows may share it,
    e.g. the two figures of one claim or the two items of one box line)."""
    counters: dict[str, int] = {}
    for r in rows:
        n = counters.get(r["product_id"], 0)
        counters[r["product_id"]] = n + 1
        r["row_id"] = f"{r['product_id']}:{table}:{n}"


def _ev_cols(store: EvidenceStore, eid: Optional[str]) -> dict:
    ev = store.get(eid) if eid else None
    if ev is None:
        return {"evidence_id": eid or "", "source_section": "", "snapshot_file": "", "snapshot_kind": "",
                "source_offset": "", "visibility": "", "evidence_tier": ""}
    return {
        "evidence_id": ev.id,
        "source_section": ev.region or ev.source_section or "",
        "snapshot_file": ev.snapshot_file or "",
        "snapshot_kind": ev.snapshot_kind or "",
        "source_offset": "" if ev.source_offset is None else ev.source_offset,
        "visibility": ev.visibility or "",
        "evidence_tier": ev.evidence_tier or "",
    }


def _topic_of(text: str) -> str:
    low = text.lower()
    if "plaque" in low:
        return "plaque"
    if "gum" in low:
        return "gums"
    if "stain" in low or "whiten" in low or "white" in low:
        return "stains/whitening"
    return "other"


def _disclaimer_map(product: NormalizedProduct) -> dict[str, list[dict]]:
    """claim text -> [{marker, text}] from the linked disclaimers."""
    out: dict[str, list[dict]] = {}
    if product.disclaimers.status == "stated":
        for d in product.disclaimers.value or []:
            for a in d.get("applies_to", []):
                out.setdefault(a["text"], []).append({"marker": d.get("marker", ""), "text": d["text"]})
    return out


_GUM_HEALTH = r"healthier gums|gum health|gingivitis|\bgums\b"


def _topic_of_segment(text: str, whole: str = "") -> str:
    """Topic named by a clause; 'along the gumline' after a plaque statement is plaque, not gum health."""
    low = text.lower()
    if re.match(r"[^a-z]*\d+(?:\.\d+)?\s*(?:x|%)\s+cleaner\b", low):
        return "cleaner teeth"  # "100% cleaner teeth and healthier gums": the figure belongs to the teeth
    if re.search(r"gumline", low.replace(" ", "")) and not re.search(_GUM_HEALTH, low):
        if "plaque" in low or "plaque" in whole.lower():
            return "plaque along gumline"
    best = None
    for name, rx in (("plaque", r"plaque"), ("gums", _GUM_HEALTH), ("stains/whitening", r"stain|whiten|white")):
        m = re.search(rx, low)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else "other"


def _first(rx: re.Pattern, text: str) -> str:
    m = rx.search(text)
    return m.group(0).strip(" ,.;*") if m else ""


def _segments(text: str) -> list[tuple[Optional[re.Match], str, str]]:
    """[(figure match, the figure's own clause, the words before it back to the previous figure)];
    one entry (None, text, "") when there is no figure."""
    figs = list(_FIGURE_RX.finditer(text))
    if not figs:
        return [(None, text, "")]
    out = []
    for i, m in enumerate(figs):
        end = figs[i + 1].start() if i + 1 < len(figs) else len(text)
        out.append((m, text[m.start():end], text[figs[i - 1].end() if i else 0: m.start()]))
    return out


def build_claims_long(products: list[NormalizedProduct], store: EvidenceStore) -> list[dict]:
    """One row per (claim text, figure): a sentence that states two figures
    ("up to 100% healthier gums** and 1000% more plaque removal*") yields two
    rows, each with the topic, comparator, timeframe and footnote of its own
    clause."""
    rows: list[dict] = []
    for p in products:
        dmap = _disclaimer_map(p)
        seen: dict[tuple, dict] = {}
        for attribute in ("claims", "plaque_removal_claim", "gum_health_claim"):
            av: AttributeValue = getattr(p, attribute)
            if av.status != "stated":
                continue
            values = av.value if isinstance(av.value, list) else [av.value]
            for i, text in enumerate(values):
                if not isinstance(text, str) or not text.strip():
                    continue
                eid = av.evidence_ids[i] if i < len(av.evidence_ids) else (av.evidence_ids[0] if av.evidence_ids else None)
                disclaimers = dmap.get(text, [])
                whole_comparator = _first(_COMPARATOR_RX, text)
                segs = _segments(text)
                for n, (fig, segment, prefix) in enumerate(segs):
                    key = (re.sub(r"\W+", " ", text.lower()).strip(), fig.group(0).lower() if fig else "")
                    if key in seen:
                        seen[key]["also_in"].add(attribute)
                        continue
                    markers = marker_runs(segment)
                    # footnotes whose marker sits on this clause (all of them for a single-figure claim)
                    mine = [d for d in disclaimers if d["marker"] in markers] or (disclaimers if len(segs) == 1 else [])
                    comparator = _first(_COMPARATOR_RX, segment) or whole_comparator
                    timeframe = _first(_TIMEFRAME_RX, segment)
                    condition = _first(_CONDITION_RX, segment)
                    for d in mine:  # the footnote often carries the comparator / timeframe / mode
                        comparator = comparator or _first(_COMPARATOR_RX, d["text"])
                        timeframe = timeframe or _first(_TIMEFRAME_RX, d["text"])
                        condition = condition or _first(_CONDITION_RX, d["text"])
                    topic = "other"
                    if fig:
                        topic = _topic_of_segment(segment, text)
                        if topic == "other":
                            topic = _topic_of_segment(prefix + segment, text)
                    else:
                        topic = _topic_of(text)
                    row = {
                        "product_id": p.product_id,
                        "claim_text": text,
                        "attribute": attribute,
                        "topic": topic,
                        "figure": fig.group(0).strip() if fig else "",
                        "figure_value": fig.group("num") if fig else "",
                        "figure_unit": fig.group("unit").lower() if fig else "",
                        "comparator": comparator,
                        "timeframe": timeframe,
                        "condition": condition,
                        "marker": ", ".join(markers),
                        "disclaimer_text": " | ".join(d["text"] for d in mine),
                        "conflict": "",
                        "also_in": set(),
                        **_ev_cols(store, eid),
                    }
                    seen[key] = row
                    rows.append(row)
    _flag_conflicts(rows)
    _inherit_headline_qualifiers(rows)
    for r in rows:
        r["also_in"] = ", ".join(sorted(r["also_in"]))
    assign_row_ids(rows, "claims")
    return rows


def _norm_figure(row: dict) -> tuple[str, str]:
    return (row["figure_value"].rstrip("0").rstrip(".") if "." in row["figure_value"] else row["figure_value"], row["figure_unit"])


def _inherit_headline_qualifiers(rows: list[dict]) -> None:
    """A claim quoted from a specification table repeats a footnote marker
    that does not reliably map to the page's footnote list (the 9900's
    'Up to 15x healthier gums***'); when a headline claim states the same
    figure for the same topic, its comparator / timeframe are used instead."""
    by_product: dict[str, list[dict]] = {}
    for r in rows:
        by_product.setdefault(r["product_id"], []).append(r)
    for rs in by_product.values():
        for r in rs:
            if not r["figure"] or not r["source_section"].startswith("Technical specifications"):
                continue
            twin = next((o for o in rs if o is not r and o["figure"] and o["topic"] == r["topic"]
                         and _norm_figure(o) == _norm_figure(r) and (o["timeframe"] or o["comparator"])), None)
            if twin:
                r["comparator"], r["timeframe"] = twin["comparator"], twin["timeframe"]
                r["condition"] = twin["condition"] or r["condition"]
                r["disclaimer_text"] = twin["disclaimer_text"]


def _flag_conflicts(rows: list[dict]) -> None:
    """Same product, same topic, different figures (the iO3 page says 6x and
    400% plaque removal along the gumline; the Philips 4100 says 700% and 5x):
    keep every row and flag them so nothing downstream picks one silently."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["figure"] and r["topic"] in ("plaque", "plaque along gumline", "gums", "stains/whitening"):
            groups.setdefault((r["product_id"], r["topic"]), []).append(r)
    for rs in groups.values():
        figures = {_norm_figure(r) for r in rs}
        if len(figures) < 2:
            continue
        for r in rs:
            others = sorted({f"{o['figure']} ({o['source_section'] or 'section ?'})" for o in rs if _norm_figure(o) != _norm_figure(r)})
            r["conflict"] = "page states other figures for the same topic: " + "; ".join(others)


# ---------------------------------------------------------------------------------------------------

# "62,000 brush movements / min", "100-240 V", "2-year limited warranty" (the hyphen joins number and unit)
_VALUE_UNIT = re.compile(r"^(?P<num>\d[\d,.]*(?:[\-/]\d[\d,.]*)?)[\s\-]*(?P<unit>[A-Za-z%°/·\. ]{1,40})?$")

# Units a specs_long row may carry; anything else after a number is part of the text, not a unit.
UNIT_WHITELIST = {
    "", "V", "W", "Hz", "mm", "cm", "g", "kg", "oz", "lb", "in", "dB", "mAh", "%",
    "days", "weeks", "hours", "minutes", "seconds", "s", "years",
    "per min", "count", "modes", "levels", "zones",
}
_UNIT_CANON = {u.lower(): u for u in UNIT_WHITELIST if u}
_UNIT_CANON.update({
    "day": "days", "week": "weeks", "hour": "hours", "hr": "hours", "hrs": "hours", "minute": "minutes",
    "min": "minutes", "mins": "minutes", "second": "seconds", "sec": "seconds", "year": "years", "yr": "years",
    "brush movements / min": "per min", "brush movements/min": "per min", "brush movements per min": "per min",
    "movements / min": "per min", "movements/min": "per min", "per minute": "per min", "/ min": "per min",
})
_UP_TO = re.compile(r"^up to\s+", re.I)
_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+$")


def _canonical_unit(unit: str) -> str:
    key = re.sub(r"\s+", " ", unit.strip().lower()).rstrip(".")
    if key in _UNIT_CANON:
        return _UNIT_CANON[key]
    if re.match(r"years?\b", key):  # "year limited warranty"
        return "years"
    return key and "?" + unit.strip()  # marks a unit that is not whitelisted


def split_value_unit(raw: str) -> tuple[str, str]:
    """(number, unit) for "62,000 brush movements / min"; the unit is a whitelisted label
    or "". Text that is not number + known unit comes back whole with no unit."""
    text = raw.strip()
    m = _VALUE_UNIT.match(text)
    if m:
        unit = _canonical_unit(m.group("unit") or "")
        if not unit.startswith("?"):
            num = m.group("num")
            return (num.replace(",", "") if _THOUSANDS.match(num) else num), unit
    return text, ""


def parse_spec_value(raw: str, group: str = "") -> dict[str, str]:
    """Normalise a spec cell without ever replacing the source text (callers keep ``raw`` as
    ``value_raw``): {value, unit, qualifier, item_text}."""
    text = raw.strip()
    qualifier = ""
    if _UP_TO.match(text):
        qualifier, text = "up_to", _UP_TO.sub("", text)
    if group == "Items included":
        m = re.match(r"(\d+)\s*(.*)$", text)
        if m:  # "1 Prestige USB charging travel case": a count of items, the words are the item
            return {"value": m.group(1), "unit": "count", "qualifier": qualifier, "item_text": m.group(2).strip()}
        return {"value": text, "unit": "", "qualifier": qualifier, "item_text": ""}
    value, unit = split_value_unit(text)
    return {"value": value, "unit": unit, "qualifier": qualifier, "item_text": ""}


_ATTR_SPECS = [
    # (group, name, attribute, unit)
    ("Dimensions and weight", "Dimensions", "dimensions", ""),
    ("Dimensions and weight", "Weight", "weight", ""),
    ("Battery", "Battery life", "battery_life_days", "days"),
    ("Battery", "Battery type", "battery_type", ""),
    ("Battery", "Charge time", "charge_time_hours", "hours"),
    ("Battery", "Battery life (as stated)", "battery_life_note", ""),
    ("Service", "Warranty", "warranty_years", "years"),
    ("Service", "Money-back guarantee", "money_back_guarantee_days", "days"),
    ("Modes", "Number of modes", "mode_count", "modes"),
    ("Modes", "Number of intensities", "intensity_count", "levels"),
    ("Timer", "Brushing zones", "brushing_zones", "zones"),
    ("Timer", "Seconds per zone", "zone_interval_seconds", "s"),
    ("Cleaning", "Brush movements per minute", "brush_movements_per_min", "per min"),
    ("Cleaning", "Cleaning technology", "cleaning_technology", ""),
    ("Cleaning", "Brush head design", "brush_head_design", ""),
    ("Cleaning", "Noise", "noise_db", "dB"),
    ("Box", "Brush heads included", "brush_heads_included", "count"),
]


def build_specs_long(products: list[NormalizedProduct], store: EvidenceStore) -> list[dict]:
    rows: list[dict] = []
    for p in products:
        drawer_names: set[str] = set()  # names already given by the Philips drawer (its raw wording wins)
        # Philips specification-table rows: "Group: Name - value"
        spec = p.specifications
        if p.brand == "Philips" and spec.status == "stated":
            for i, text in enumerate(spec.value or []):
                m = re.match(r"(?P<group>[^:]+):\s*(?P<name>.+?)\s+-\s+(?P<value>.+)$", text)
                if not m:
                    continue
                parsed = parse_spec_value(m.group("value"), m.group("group").strip())
                eid = spec.evidence_ids[i] if i < len(spec.evidence_ids) else None
                drawer_names.add(m.group("name").strip().lower())
                rows.append({"product_id": p.product_id, "group": m.group("group"), "name": m.group("name"),
                             "value": parsed["value"], "unit": parsed["unit"], "qualifier": parsed["qualifier"],
                             "item_text": parsed["item_text"], "value_raw": m.group("value"), "origin": "drawer",
                             **_ev_cols(store, eid)})
        for group, name, attribute, unit in _ATTR_SPECS:
            av: AttributeValue = getattr(p, attribute)
            if av.status != "stated" or av.value in (None, "", []):
                continue
            if name.lower() in drawer_names:
                continue  # one row per fact: "Warranty", "Battery type" come from the drawer
            raw = str(av.value)
            value, parsed_unit = (raw, "") if not isinstance(av.value, (int, float)) else (raw, unit)
            rows.append({"product_id": p.product_id, "group": group, "name": name, "value": value,
                         "unit": parsed_unit or unit, "qualifier": "", "item_text": "", "value_raw": av.raw_value or raw,
                         "origin": "derived", **_ev_cols(store, av.evidence_ids[0] if av.evidence_ids else None)})
    assign_row_ids(rows, "specs")
    return rows


def clean_headline(text: str) -> str:
    """Strip extraction artefacts: literal backslash-n or real newlines, trailing
    dashes/colons, repeated spaces."""
    text = html.unescape(text or "")
    text = re.sub(r"\\+n|[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" -–—:\\").strip()


# brand slogans, not product features: kept (kind = brand_filler) so retrieval can down-weight them
_BRAND_FILLER_RX = re.compile(
    r"^(?:switch to oral.b|trust oral.b|make the most of every session|oral.b (?:is|has|gives)\b)|\bcarefree$|with confidence$",
    re.I,
)


def build_feature_headlines(records_by_id: dict[str, dict]) -> list[dict]:
    """Marketing headlines: cleaned, one-word fragments dropped ("app"), deduplicated."""
    rows: list[dict] = []
    for pid, record in records_by_id.items():
        d = record.get("details") or {}
        seen: set[str] = set()
        counter = [0]

        def add(text: str, kind: str, where: str) -> None:
            text = clean_headline(text)
            key = text.lower()
            if len(text.split()) < 2 or key in seen:
                return
            seen.add(key)
            if _BRAND_FILLER_RX.search(text):
                kind = "brand_filler"
            rows.append({"product_id": pid, "evidence_id": f"{pid}:feature_headline:{counter[0]}", "headline": text,
                         "kind": kind, "source_section": where})
            counter[0] += 1

        if record["brand"] == "Oral-B":
            for term in d.get("key_terms") or []:
                add(term, "bullet_leadin", "Product description")
        else:
            for f in d.get("features") or []:
                add(f["name"], "feature_panel", "Feature panels")
            for h in d.get("highlights") or []:
                add(h, "highlight", "Feature panels")
    assign_row_ids(rows, "feature_headlines")
    return rows


def build_included_items_long(products: list[NormalizedProduct], store: EvidenceStore) -> list[dict]:
    """One row per box item; a line that lists several items ("1 X; 1 Y") yields one row each."""
    from .included import parse_included

    rows: list[dict] = []
    for p in products:
        av = p.included_items
        if av.status != "stated":
            continue
        for i, text in enumerate(av.value or []):
            eid = av.evidence_ids[i] if i < len(av.evidence_ids) else None
            for entry in parse_included(text):
                rows.append({"product_id": p.product_id, "item": entry["item"].strip(), "item_raw": text,
                             "quantity": entry["quantity"], "quantity_source": entry["quantity_source"],
                             "kind": entry["kind"], **_ev_cols(store, eid)})
    assign_row_ids(rows, "included_items")
    return rows
