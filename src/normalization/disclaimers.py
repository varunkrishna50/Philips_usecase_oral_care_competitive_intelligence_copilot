"""Link page disclaimers (footnotes) to the statements they qualify.

Pages mark a qualified statement with a run of footnote symbols
("...20x more effective on plaque*, gentle on gums") and list the footnote
elsewhere ("* vs. a manual toothbrush"). A disclaimer belongs to a
statement when the statement carries a marker run of exactly the same
length ("*" is not "**"). Footnotes that are scoped to one page section
(e.g. a warranty-tab footnote) only attach to statements from that section.

This module is pure matching logic; evidence records for the disclaimer
texts are created by detail_attributes.apply_detail_attributes.
"""
from __future__ import annotations

import re
from typing import Any

_MARKER_RUN = re.compile(r"(?<!\*)\*+(?!\*)|[†‡§]+")


def marker_runs(text: str) -> list[str]:
    """Distinct footnote-symbol runs in a statement, in order of appearance."""
    runs: list[str] = []
    for m in _MARKER_RUN.finditer(text):
        if m.group(0) not in runs:
            runs.append(m.group(0))
    return runs


def link_disclaimers(disclaimers: list[dict], statements: list[dict]) -> dict[str, Any]:
    """
    disclaimers: [{marker, text, source, section}]
    statements:  [{attribute, text, section}]  (section optional)

    Returns {
      "disclaimers": [{marker, text, source, section, applies_to: [{attribute, text}]}],
      "per_statement": {text: [disclaimer texts]},
      "orphan_disclaimers": [text, ...]            # nothing carries this marker
      "ambiguous_markers": [marker, ...]           # one marker, several disclaimers
      "markers_without_disclaimer": [(text, marker), ...],
    }
    """
    by_marker: dict[str, list[dict]] = {}
    for d in disclaimers:
        by_marker.setdefault(d.get("marker") or "", []).append(d)

    linked = [dict(d, applies_to=[]) for d in disclaimers]
    per_statement: dict[str, list[str]] = {}
    markers_without: list[tuple[str, str]] = []

    seen_statement: set[str] = set()
    for st in statements:
        text, attribute = st["text"], st["attribute"]
        # The same sentence can be reached through several attributes
        # (claims, a page section, ...); report it once, under the first.
        if text in seen_statement:
            continue
        seen_statement.add(text)
        for run in marker_runs(text):
            matched = False
            for d in linked:
                if (d.get("marker") or "") != run:
                    continue
                # A footnote printed in one page section only qualifies statements
                # from that section (e.g. the warranty tab's "*" is not the
                # description's "*").
                if d.get("section") and st.get("section") and st["section"] != d["section"]:
                    continue
                matched = True
                if not any(a["text"] == text and a["attribute"] == attribute for a in d["applies_to"]):
                    d["applies_to"].append({"attribute": attribute, "text": text})
                bucket = per_statement.setdefault(text, [])
                if d["text"] not in bucket:
                    bucket.append(d["text"])
            if not matched:
                markers_without.append((text, run))

    orphans = [d["text"] for d in linked if not d["applies_to"]]
    ambiguous = sorted(
        {m for m, ds in by_marker.items() if m and len([d for d in ds if d.get("source") != "section"]) > 1}
    )
    return {
        "disclaimers": linked,
        "per_statement": per_statement,
        "orphan_disclaimers": orphans,
        "ambiguous_markers": ambiguous,
        "markers_without_disclaimer": markers_without,
    }
