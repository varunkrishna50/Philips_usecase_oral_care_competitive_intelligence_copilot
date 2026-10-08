"""Independent, mechanical validation of the normalized output.

This re-derives everything from data/raw/ itself (never trusting the
in-memory objects normalize.py already built) so a bug in extraction
can't also hide itself from validation. Any failure here means the build
should NOT be treated as successful.
"""
from __future__ import annotations

from pathlib import Path

from .evidence import EvidenceStore, resolve_source_text
from .models import ATTRIBUTE_FIELDS, EvidenceRef, NormalizedProduct


def validate(
    products: list[NormalizedProduct],
    store: EvidenceStore,
    raw_index: dict[str, dict],
) -> list[str]:
    errors: list[str] = []
    evidence = store.all()

    errors.extend(_validate_product_identity(products))
    errors.extend(_validate_evidence_integrity(products, evidence))
    errors.extend(_validate_evidence_verbatim(evidence, raw_index))
    errors.extend(_validate_no_cross_product_evidence(products, evidence))

    return errors


def _validate_product_identity(products: list[NormalizedProduct]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_records: set[tuple] = set()

    for p in products:
        if not p.product_id:
            errors.append(f"product with empty product_id (name={p.product_name!r})")
        elif p.product_id in seen_ids:
            errors.append(f"duplicate product_id: {p.product_id!r}")
        seen_ids.add(p.product_id)

        if not p.brand:
            errors.append(f"{p.product_id}: missing brand")
        if not p.product_name:
            errors.append(f"{p.product_id}: missing product_name")
        if not p.product_url:
            errors.append(f"{p.product_id}: missing product_url")

        if p.price.status == "stated" and not isinstance(p.price.value, (int, float)):
            errors.append(f"{p.product_id}: price.status is 'stated' but value is not numeric: {p.price.value!r}")
        if p.mode_count.status == "stated" and p.mode_count.value is not None:
            if not isinstance(p.mode_count.value, int) or p.mode_count.value < 0:
                errors.append(f"{p.product_id}: mode_count value must be a non-negative int, got {p.mode_count.value!r}")
        for bool_field in ("pressure_sensing", "app_connectivity", "bluetooth_connectivity", "timer"):
            av = getattr(p, bool_field)
            if av.value is not None and not isinstance(av.value, bool):
                errors.append(f"{p.product_id}: {bool_field} value must be true/false/null, got {av.value!r}")

        dedup_key = (p.brand, p.product_name, p.product_url)
        if dedup_key in seen_records:
            errors.append(f"duplicate normalized product record for {dedup_key}")
        seen_records.add(dedup_key)

    return errors


def _validate_evidence_integrity(products: list[NormalizedProduct], evidence: dict[str, EvidenceRef]) -> list[str]:
    errors: list[str] = []
    for p in products:
        for field_name in ATTRIBUTE_FIELDS:
            av = getattr(p, field_name)
            if av.status == "not_mentioned" and av.evidence_ids:
                errors.append(f"{p.product_id}.{field_name}: status is not_mentioned but has evidence_ids {av.evidence_ids}")
            if av.status != "not_mentioned" and not av.evidence_ids:
                errors.append(f"{p.product_id}.{field_name}: status is {av.status!r} but has no evidence_ids")
            for eid in av.evidence_ids:
                ev = evidence.get(eid)
                if ev is None:
                    errors.append(f"{p.product_id}.{field_name}: evidence id {eid!r} does not exist in evidence.json")
                    continue
                if not ev.evidence_text:
                    errors.append(f"{eid}: empty evidence_text")
                if not ev.source_url:
                    errors.append(f"{eid}: empty source_url")
                if not ev.source_file:
                    errors.append(f"{eid}: empty source_file")
    return errors


def _validate_evidence_verbatim(evidence: dict[str, EvidenceRef], raw_index: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    for eid, ev in evidence.items():
        record = raw_index.get(ev.product_id)
        if record is None:
            errors.append(f"{eid}: no raw record found for product_id {ev.product_id!r}")
            continue
        html_path = Path(record["raw_html_path"])
        try:
            source_text = resolve_source_text(record, ev.source_section, ev.source_section_index, html_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{eid}: could not resolve claimed source ({exc})")
            continue
        if ev.evidence_text not in source_text:
            errors.append(
                f"{eid}: evidence_text {ev.evidence_text!r} is NOT a verbatim substring of its claimed "
                f"source (section={ev.source_section!r}, index={ev.source_section_index})"
            )
        # Note: an out-of-range source_section_index or a source_section
        # that doesn't match the record's actual heading at that index is
        # already caught above -- resolve_source_text() raises ValueError
        # for both, which the except clause turns into a "could not
        # resolve claimed source" error.
    return errors


def _validate_no_cross_product_evidence(products: list[NormalizedProduct], evidence: dict[str, EvidenceRef]) -> list[str]:
    errors: list[str] = []
    url_by_product_id = {p.product_id: p.product_url for p in products}
    for eid, ev in evidence.items():
        expected_url = url_by_product_id.get(ev.product_id)
        if expected_url is None:
            errors.append(f"{eid}: evidence references unknown product_id {ev.product_id!r}")
        elif ev.source_url != expected_url:
            errors.append(
                f"{eid}: source_url {ev.source_url!r} does not match {ev.product_id}'s own "
                f"product_url {expected_url!r} -- possible cross-product evidence"
            )
    return errors
