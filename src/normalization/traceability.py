"""Evidence traceability: where each verbatim quote lives in a saved text
snapshot, and how citable it is.

Two snapshot kinds, both plain text files on disk:

* ``sections``   - ``data/raw/text/<brand>/<id>.txt`` written by Step 1: the
  exact normalized section text the extractors saw. Evidence taken from page
  sections gets its offset measured here (never against raw HTML).
* ``structured`` - ``data/processed/snapshots/<id>.structured.txt`` written
  here: the visible text blob plus the rendered/embedded regions used for
  evidence that is not a plain section (Philips specification rows and
  feature panels, JSON-LD, rating widget, ...). Each region is labelled with
  its tier:

    visible / expanded -> may be cited
    cross_check_only   -> kept and flagged, never cited (JSON-LD, embedded price)

Philips specification/feature JSON is ``expanded`` only if the one-time
rendering gate (``scripts/verify_rendering_gate.py`` ->
``data/raw/rendering_gate.json``) confirmed that opening the toggle renders
those strings.

The validator re-opens the snapshot files and checks that
``text[offset:offset+len(quote)] == quote`` exactly.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Optional

from acquisition.snapshot import read_snapshot, render_sections

from .detail_attributes import _Page
from .evidence import EvidenceStore

CITED_TIERS = {"visible", "expanded"}


def rendering_gate(raw_dir: Path) -> bool:
    path = raw_dir / "rendering_gate.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("philips_spec_expanded"))
    except (ValueError, OSError):
        return False


def structured_regions(page: _Page, spec_tier: str) -> list[dict]:
    """[{label, tier, text}] - the text blobs structured evidence may come from."""
    regions: list[dict] = [{"label": "visible_text", "tier": "visible", "text": page.raw_text}]
    if page.widget_text:
        regions.append({"label": "rating_widget", "tier": "visible", "text": page.widget_text})
    if page.rendered:
        regions.append({"label": "rendered_text", "tier": "visible", "text": page.rendered})
    if page.brand == "Philips":
        for cname, (_obj, region) in page.chapters.items():
            regions.append({"label": f"spec:{cname}", "tier": spec_tier, "text": region})
        features = getattr(page, "features_region", None)
        if features:
            regions.append({"label": "feature_panels", "tier": spec_tier, "text": features})
        stem = Path(page.record["raw_html_path"]).stem
        sku = stem.split("_", 1)[1].upper().replace("-", "/")
        at = page.unescaped.find('"baseCommerceProduct":{"id":"' + sku + '"')
        if at >= 0:
            regions.append({"label": "embedded_price", "tier": "cross_check_only", "text": page.unescaped[at: at + 3000]})
    labels = []
    for color in (page.details.get("colors") or []):
        m = re.search(r'alt="[^"]*' + re.escape(color) + r'[^"]*"', page.html, re.I)
        if m and m.group(0) not in labels:
            labels.append(m.group(0))
    if labels:
        regions.append({"label": "swatch_labels", "tier": "visible", "text": "\n".join(labels)})
    lds = re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', page.html, re.S)
    if lds:
        regions.append({"label": "json_ld", "tier": "cross_check_only", "text": "\n".join(lds)})
    metas = re.findall(r'<meta property="product:price:[^>]*>', page.html)
    if metas:
        regions.append({"label": "meta_price", "tier": "cross_check_only", "text": "\n".join(metas)})
    return regions


def _render_structured(regions: list[dict]) -> tuple[str, list[dict]]:
    parts, spans, pos = [], [], 0
    for r in regions:
        header = f"### {r['label']} [{r['tier']}]\n"
        spans.append({**r, "text_start": pos + len(header), "text_end": pos + len(header) + len(r["text"])})
        block = header + r["text"] + "\n\n"
        parts.append(block)
        pos += len(block)
    return "".join(parts), spans


_GENERIC_REGIONS = {"visible_text", "rendered_text"}


def _real_section(page: _Page, record: dict, text: str, label: str) -> str:
    """Evidence located in the whole-page text blobs is attributed to the page
    section (or specification row) that actually holds the quote."""
    if label not in _GENERIC_REGIONS:
        return label
    probe = text.rstrip("*").strip()
    for section in record["sections"]:
        if probe and (probe in section["text"] or probe in section["heading"]):
            return section["heading"]
    if page.brand == "Philips":
        for cname, (obj, _region) in page.chapters.items():
            for item in obj.get("csItem", []):
                iname = html.unescape(str(item.get("csItemName", "")))
                values = [html.unescape(str(v.get("csValueName", ""))).strip() for v in item.get("csValue", [])]
                if probe == iname or probe in values or probe in [v.rstrip("*").strip() for v in values]:
                    return f"Technical specifications > {cname} > {iname}"
    return "Product page (" + ("rendered text" if label == "rendered_text" else "page text") + ")"


def annotate_evidence(store: EvidenceStore, records_by_id: dict[str, dict], raw_dir: Path, snapshot_dir: Path) -> dict:
    """Fill snapshot_file / snapshot_kind / source_offset / region / visibility /
    evidence_tier on every evidence record and write the structured snapshots.
    Returns counts for the build summary."""
    gate = rendering_gate(raw_dir)
    spec_tier = "expanded" if gate else "cross_check_only"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    counts = {"located": 0, "missing_offset": 0, "cross_check_only": 0}

    section_cache: dict[str, tuple[str, list[dict]]] = {}
    structured_cache: dict[str, tuple[str, list[dict], str]] = {}

    for ev in store.all().values():
        record = records_by_id.get(ev.product_id)
        if record is None:
            continue
        pid = ev.product_id

        # ---- section snapshot --------------------------------------------------
        if pid not in section_cache:
            snap = record.get("text_snapshot_path")
            if snap and Path(snap).exists():
                text = read_snapshot(Path(snap))
            else:
                text, _ = render_sections(record["sections"])
            section_cache[pid] = (text, render_sections(record["sections"])[1])
        sec_text, sec_spans = section_cache[pid]

        located = False
        if ev.source_section_index is not None and ev.source_section != "structured_data":
            span = sec_spans[ev.source_section_index] if ev.source_section_index < len(sec_spans) else None
            if span:
                body = sec_text[span["text_start"]: span["text_end"]]
                at = body.find(ev.evidence_text)
                if at >= 0:
                    ev.source_offset = span["text_start"] + at
                    located = True
                else:
                    heading = sec_text[span["heading_start"]: span["heading_start"] + len(span["heading"])]
                    at = heading.find(ev.evidence_text)
                    if at >= 0:
                        ev.source_offset = span["heading_start"] + at
                        located = True
                if located:
                    ev.snapshot_file = record.get("text_snapshot_path")
                    ev.snapshot_kind = "sections"
                    ev.region = span["heading"]
                    section = record["sections"][ev.source_section_index]
                    ev.visibility = section.get("visibility", "visible")
                    ev.evidence_tier = "cited"
        if located:
            counts["located"] += 1
            continue

        # ---- structured snapshot -----------------------------------------------------
        if pid not in structured_cache:
            page = _Page(record)
            text, spans = _render_structured(structured_regions(page, spec_tier))
            path = snapshot_dir / f"{pid}.structured.txt"
            with path.open("w", encoding="utf-8", newline="") as f:
                f.write(text)
            rel = path.resolve().relative_to(raw_dir.resolve().parents[1]) if raw_dir.resolve().parents[1] in path.resolve().parents else path
            structured_cache[pid] = (text, spans, str(rel), page)
        s_text, s_spans, s_path, s_page = structured_cache[pid]
        for span in s_spans:
            body = s_text[span["text_start"]: span["text_end"]]
            at = body.find(ev.evidence_text)
            if at >= 0:
                ev.source_offset = span["text_start"] + at
                ev.snapshot_file = s_path.replace("\\", "/")
                ev.snapshot_kind = "structured"
                ev.region = _real_section(s_page, record, ev.evidence_text, span["label"])
                ev.visibility = span["tier"] if span["tier"] in CITED_TIERS else "hidden"
                ev.evidence_tier = "cited" if span["tier"] in CITED_TIERS else "cross_check_only"
                located = True
                break
        if located:
            counts["located"] += 1
            if ev.evidence_tier == "cross_check_only":
                counts["cross_check_only"] += 1
        else:
            counts["missing_offset"] += 1
    return counts


def annotate_headlines(rows: list[dict], records_by_id: dict[str, dict], raw_dir: Path, snapshot_dir: Path) -> dict:
    """Provenance for ``feature_headlines`` rows, kept in that file only (the
    headlines are not ``evidence.json`` records, so the retrieval corpus is
    unaffected): the section snapshot first, then the structured snapshot, with
    the same exact-substring rule as ``annotate_evidence``. The validator
    re-opens the snapshot and checks the headline sits at ``source_offset``."""
    spec_tier = "expanded" if rendering_gate(raw_dir) else "cross_check_only"
    root = raw_dir.resolve().parents[1]
    sections: dict[str, tuple[str, list[dict]]] = {}
    structured: dict[str, tuple[str, list[dict], str, _Page]] = {}
    counts = {"located": 0, "missing": 0}
    for row in rows:
        row.update({"snapshot_file": "", "snapshot_kind": "", "source_offset": "", "visibility": "", "evidence_tier": "",
                    "quote_form": ""})
        record = records_by_id.get(row["product_id"])
        if record is None:
            counts["missing"] += 1
            continue
        pid, text = row["product_id"], row["headline"]
        if pid not in sections:
            snap = record.get("text_snapshot_path")
            body = read_snapshot(Path(snap)) if snap and Path(snap).exists() else render_sections(record["sections"])[0]
            sections[pid] = (body, render_sections(record["sections"])[1])
        sec_text, spans = sections[pid]
        located = False
        for i, span in enumerate(spans):
            at = sec_text[span["text_start"]: span["text_end"]].find(text)
            offset = span["text_start"] + at if at >= 0 else None
            if offset is None:
                at = sec_text[span["heading_start"]: span["heading_start"] + len(span["heading"])].find(text)
                offset = span["heading_start"] + at if at >= 0 else None
            if offset is not None:
                row.update(snapshot_file=record.get("text_snapshot_path") or "", snapshot_kind="sections",
                           source_offset=offset, source_section=span["heading"],
                           visibility=record["sections"][i].get("visibility", "visible"), evidence_tier="cited")
                located = True
                break
        if not located:
            if pid not in structured:
                page = _Page(record)
                s_text, s_spans = _render_structured(structured_regions(page, spec_tier))
                path = snapshot_dir / f"{pid}.structured.txt"
                rel = path.resolve().relative_to(root) if root in path.resolve().parents else path
                structured[pid] = (s_text, s_spans, str(rel).replace("\\", "/"), page)
            s_text, s_spans, s_path, page = structured[pid]
            for span in s_spans:
                at = s_text[span["text_start"]: span["text_end"]].find(text)
                if at >= 0:
                    cited = span["tier"] in CITED_TIERS
                    row.update(snapshot_file=s_path, snapshot_kind="structured", source_offset=span["text_start"] + at,
                               source_section=_real_section(page, record, text, span["label"]),
                               visibility=span["tier"] if cited else "hidden",
                               evidence_tier="cited" if cited else "cross_check_only")
                    located = True
                    break
        if not located and "&" in text:
            # The panel title exists only in the page's JSON ("SmarTimer \u0026 BrushPacer"), not in the
            # rendered text: keep the pointer but do not cite it.
            esc = text.replace("&", "\\u0026")
            s_text, s_spans, s_path, page = structured[pid]
            for span in s_spans:
                at = s_text[span["text_start"]: span["text_end"]].find(esc)
                if at >= 0:
                    row.update(snapshot_file=s_path, snapshot_kind="structured", source_offset=span["text_start"] + at,
                               source_section=_real_section(page, record, text, span["label"]), visibility="hidden",
                               evidence_tier="cross_check_only", quote_form="json_escaped")
                    located = True
                    break
        counts["located" if located else "missing"] += 1
    return counts
