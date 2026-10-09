"""Cleaned text snapshots.

Step 1 saves, per product, the exact normalized text the extractors saw
(``data/raw/text/<brand>/<product_id>.txt``). Evidence offsets are computed
against that file - never against raw HTML - so whitespace- or
quote-normalization in the section text cannot make a quote "disappear".

File layout, one block per section::

    === [<index>] <heading> ===
    <section text, one line per block element>
    <blank line>

``render_sections`` is the single place that defines this layout; the
writer (Step 1) and the offset calculator (Step 2) both call it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


def render_sections(sections: Iterable[Mapping]) -> tuple[str, list[dict]]:
    """Return (text, spans). ``spans[i]`` = {index, heading, header_start,
    text_start, text_end} in character offsets into ``text``."""
    parts: list[str] = []
    spans: list[dict] = []
    pos = 0
    for index, section in enumerate(sections):
        heading = str(section["heading"])
        body = str(section["text"])
        header = f"=== [{index}] {heading} ===\n"
        spans.append(
            {
                "index": index,
                "heading": heading,
                "header_start": pos,
                "heading_start": pos + len(f"=== [{index}] "),
                "text_start": pos + len(header),
                "text_end": pos + len(header) + len(body),
            }
        )
        block = header + body + "\n\n"
        parts.append(block)
        pos += len(block)
    return "".join(parts), spans


def snapshot_path(raw_dir: Path, brand_key: str, product_id: str) -> Path:
    """``raw_dir`` is ``<repo>/data/raw``."""
    return raw_dir / "text" / brand_key / f"{product_id}.txt"


def write_snapshot(path: Path, sections: Iterable[Mapping]) -> None:
    text, _ = render_sections(sections)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps offsets identical on every platform (no \n -> \r\n).
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def read_snapshot(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.read()
