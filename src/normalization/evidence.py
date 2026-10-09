"""Evidence collection and verbatim-source resolution.

Every EvidenceRef is created through EvidenceStore.add(), which assigns a
deterministic id ("{product_id}:{attribute}:{n}", n starting at 0 per
product+attribute pair) and -- as a first line of defense -- asserts the
evidence text is actually a substring of the text it claims to come from
before accepting it. validation.py performs an independent second check
of the same property, re-reading straight from data/raw/, as the
mechanical quality gate described in the Step 2 spec.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .models import EvidenceRef, Status

# Sentinel used for source_section when evidence comes from structured_data
# (JSON-LD / price_text) rather than from a specific {heading, text} section.
STRUCTURED_DATA_SENTINEL = "structured_data"


class EvidenceIntegrityError(ValueError):
    """Raised when code under our control tries to create evidence whose
    text is not a verbatim substring of its claimed source."""


def structured_source_text(raw_text: str, html_text: str, extra_text: str = "") -> str:
    """Everything structured-data evidence may be quoted from: the visible
    text, the raw HTML, the HTML with one level of JSON string escaping
    removed (Philips embeds its product JSON as backslash-escaped JS
    strings, so quoting it unescaped keeps evidence readable), and any
    text captured from a rendered widget (extra_text)."""
    unescaped = html_text.replace('\\"', '"')
    return "\n".join((raw_text, html_text, unescaped, extra_text))


class EvidenceStore:
    def __init__(self) -> None:
        self._items: dict[str, EvidenceRef] = {}
        self._counters: dict[tuple[str, str], int] = {}

    def add(
        self,
        *,
        product_id: str,
        attribute: str,
        value: Any,
        status: Status,
        raw_value: Optional[str],
        evidence_text: str,
        source_url: str,
        source_file: str,
        source_section: Optional[str],
        source_section_index: Optional[int],
        source_text_for_check: str,
    ) -> str:
        if evidence_text not in source_text_for_check:
            raise EvidenceIntegrityError(
                f"evidence_text {evidence_text!r} for {product_id}:{attribute} "
                "is not a verbatim substring of its claimed source"
            )
        key = (product_id, attribute)
        n = self._counters.get(key, 0)
        self._counters[key] = n + 1
        eid = f"{product_id}:{attribute}:{n}"
        self._items[eid] = EvidenceRef(
            id=eid,
            product_id=product_id,
            attribute=attribute,
            value=value,
            status=status,
            raw_value=raw_value,
            evidence_text=evidence_text,
            source_url=source_url,
            source_file=source_file,
            source_section=source_section,
            source_section_index=source_section_index,
        )
        return eid

    def add_from_section(
        self,
        *,
        product_id: str,
        attribute: str,
        value: Any,
        status: Status,
        raw_value: Optional[str],
        evidence_text: str,
        source_url: str,
        source_file: str,
        section_heading: str,
        section_index: int,
        section_combined_text: str,
    ) -> str:
        return self.add(
            product_id=product_id,
            attribute=attribute,
            value=value,
            status=status,
            raw_value=raw_value,
            evidence_text=evidence_text,
            source_url=source_url,
            source_file=source_file,
            source_section=section_heading,
            source_section_index=section_index,
            source_text_for_check=section_combined_text,
        )

    def add_from_structured(
        self,
        *,
        product_id: str,
        attribute: str,
        value: Any,
        status: Status,
        raw_value: Optional[str],
        evidence_text: str,
        source_url: str,
        source_file: str,
        raw_text: str,
        html_text: str,
        extra_text: str = "",
    ) -> str:
        return self.add(
            product_id=product_id,
            attribute=attribute,
            value=value,
            status=status,
            raw_value=raw_value,
            evidence_text=evidence_text,
            source_url=source_url,
            source_file=source_file,
            source_section=STRUCTURED_DATA_SENTINEL,
            source_section_index=None,
            source_text_for_check=structured_source_text(raw_text, html_text, extra_text),
        )

    def discard(self, evidence_id: str) -> None:
        """Remove an evidence record that no attribute will reference any
        more (e.g. a value superseded by a better-sourced one), so it never
        reaches evidence.json / the retrieval index. If it was the most
        recently issued id for its product+attribute the counter is rolled
        back so ids stay contiguous."""
        ev = self._items.pop(evidence_id, None)
        if ev is None:
            return
        key = (ev.product_id, ev.attribute)
        n = int(evidence_id.rsplit(":", 1)[1])
        if self._counters.get(key) == n + 1:
            self._counters[key] = n

    def get(self, evidence_id: str) -> Optional[EvidenceRef]:
        return self._items.get(evidence_id)

    def all(self) -> dict[str, EvidenceRef]:
        return dict(self._items)

    def to_json_list(self) -> list[dict]:
        return [ev.model_dump() for ev in self._items.values()]


def resolve_source_text(
    record: dict,
    source_section: Optional[str],
    source_section_index: Optional[int],
    html_path: Path,
) -> str:
    """Reconstruct the exact text an evidence item claims to come from,
    straight from the raw Step 1 record -- used both at evidence-creation
    time and, independently, by validation.py.
    """
    if source_section is None:
        return record["raw_text"]

    if source_section == STRUCTURED_DATA_SENTINEL:
        html_text = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
        extra = record.get("structured_data", {}).get("rating_widget_text", "")
        rendered = record.get("rendered_text_path")
        if rendered and Path(rendered).exists():
            extra += "\n" + Path(rendered).read_text(encoding="utf-8", newline="")
        return structured_source_text(record["raw_text"], html_text, extra)

    sections = record.get("sections", [])
    if source_section_index is None or not (0 <= source_section_index < len(sections)):
        raise ValueError(f"source_section_index {source_section_index!r} out of range")
    section = sections[source_section_index]
    if section["heading"] != source_section:
        raise ValueError(
            f"section heading mismatch: evidence claims {source_section!r}, "
            f"record has {section['heading']!r} at index {source_section_index}"
        )
    # A claim is frequently carried in the heading itself (e.g. Philips's
    # "Pressure Sensor helps protect your teeth and gums"), which Step 1
    # stores separately from the section body -- concatenate both so
    # evidence sourced from either validates consistently.
    return section["heading"] + "\n" + section["text"]
