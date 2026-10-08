"""Step 2 orchestration: load Step 1's raw records, dispatch per brand,
produce normalized products + the shared evidence store + warnings.

This module never touches the network -- it only reads data/raw/.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import oralb, philips
from .evidence import EvidenceStore
from .models import NormalizationWarning, NormalizedProduct

BRAND_NORMALIZERS = {
    "Philips": philips.normalize_product,
    "Oral-B": oralb.normalize_product,
}


def load_raw_records(raw_dir: Path) -> list[dict]:
    """Reads every per-product JSON record Step 1 wrote under
    data/raw/<brand>/*.json. Does not touch data/raw/html/ directly here
    (individual extractors read the HTML file only when they need it, via
    raw_html_path) and never re-scrapes anything.
    """
    records: list[dict] = []
    for brand_dir in ("philips", "oralb"):
        brand_path = raw_dir / brand_dir
        if not brand_path.exists():
            continue
        for path in sorted(brand_path.glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    records.sort(key=lambda r: Path(r["raw_html_path"]).stem)
    return records


def normalize_all(
    records: list[dict],
) -> tuple[list[NormalizedProduct], EvidenceStore, list[NormalizationWarning]]:
    store = EvidenceStore()
    warnings: list[NormalizationWarning] = []
    products: list[NormalizedProduct] = []

    for record in records:
        normalizer = BRAND_NORMALIZERS.get(record["brand"])
        if normalizer is None:
            raise ValueError(f"No normalizer registered for brand {record['brand']!r}")
        products.append(normalizer(record, store, warnings))

    return products, store, warnings


def index_raw_records(records: list[dict]) -> dict[str, dict]:
    """product_id -> raw record, for validation.py's independent re-check."""
    return {Path(r["raw_html_path"]).stem: r for r in records}
