"""Parse "What's in the box" lines into items with a quantity and a kind.

One helper feeds every place that reports box contents, so they cannot
disagree: ``included_items_long.csv``, the wide ``brush_heads_included``
column and the ``Box / Brush heads included`` row of ``specs_long.csv``.

Line shapes seen on the two sites:

* Philips   ``Brush head: 1 C3 Premium Plaque Control; 1 S2 Sensitive``
            (one label, several items, each with its own leading quantity)
            ``Handle: 1 6500 rechargeable toothbrush`` / ``Quick start guide: 1``
* Oral-B    ``(2) Brush Heads`` / ``Charger`` / ``Gum & Sensitive Replacement Brush Head``
"""
from __future__ import annotations

import re
from typing import Any, Optional

# "; 1 S2 Sensitive": a semicolon followed by a quantity starts a new item
_ITEM_SPLIT = re.compile(r";\s*(?=\d+\s)")

_LABEL_KIND = {
    "handle": "handle",
    "brush head": "brush_head",
    "brush heads": "brush_head",
    "charger": "charger",
    "travel case": "travel_case",
    "accessory": "accessory",
    "battery": "battery",
}

_KIND_RX = [
    ("brush_head", re.compile(r"brush[- ]?heads?|\brefill", re.I)),
    ("travel_case", re.compile(r"travel (?:case|pouch)|\bcase\b|pouch", re.I)),
    ("charger", re.compile(r"charger|charging (?:base|station|stand)|\bbase\b", re.I)),
    ("accessory", re.compile(r"\b(?:stand|holder|mount|cover|cap)\b", re.I)),
    ("handle", re.compile(r"\bhandles?\b|toothbrush", re.I)),
]


def classify(text: str, label: Optional[str] = None) -> str:
    """Kind of a box item. The row label wins ("Accessory: 1 Toothbrush stand"
    is an accessory); keywords are the fallback for unlabelled Oral-B lines."""
    if label:
        kind = _LABEL_KIND.get(re.sub(r"\s+", " ", label.strip().lower()))
        if kind:
            return kind
    return next((k for k, rx in _KIND_RX if rx.search(text)), "other")


def _entry(item: str, quantity: Any, kind: str, raw: str) -> dict[str, Any]:
    """``quantity_source`` says whether the page stated the number or we assumed one."""
    if quantity == "":
        return {"item": item, "quantity": 1, "quantity_source": "assumed_default", "kind": kind, "raw": raw}
    return {"item": item, "quantity": quantity, "quantity_source": "stated", "kind": kind, "raw": raw}


def parse_included(text: str) -> list[dict[str, Any]]:
    """One box line -> [{item, quantity, quantity_source, kind, raw}]. A line without a
    stated quantity gets quantity 1 and quantity_source "assumed_default"."""
    raw = text.strip()
    m = re.match(r"\s*\((\d+)\)\s*(.+)$", raw)  # Oral-B "(2) Brush Heads"
    if m:
        name = m.group(2).strip()
        return [_entry(name, int(m.group(1)), classify(name), raw)]

    m = re.match(r"(?P<label>[^:]+):\s*(?P<rest>.*)$", raw)  # Philips "Label: 1 name; 1 name"
    if m and re.match(r"\d", m.group("rest").strip() or "x"):
        label, rest = m.group("label").strip(), m.group("rest").strip()
        items = []
        for part in _ITEM_SPLIT.split(rest):
            pm = re.match(r"(?P<n>\d+)\s*(?P<name>.*)$", part.strip())
            if not pm:
                continue
            name = pm.group("name").strip() or label
            items.append(_entry(name, int(pm.group("n")), classify(name, label), raw))
        if items:
            return items

    return [_entry(raw, "", classify(raw), raw)]


def parse_all(texts: list[str]) -> list[dict[str, Any]]:
    return [entry for text in texts for entry in parse_included(text)]


def brush_head_total(texts: list[str]) -> Optional[int]:
    """Number of brush heads in the box: the sum of stated quantities; one per
    head line when no head line states a quantity (Oral-B iO2 lists two
    head lines); None when the lines mix both (no guessing)."""
    heads = [e for e in parse_all(texts) if e["kind"] == "brush_head"]
    if not heads:
        return None
    sources = {e["quantity_source"] for e in heads}
    if sources == {"stated"}:
        return sum(e["quantity"] for e in heads)
    if sources == {"assumed_default"}:
        return len(heads)
    return None
