"""Loads Step 2's frozen output into two corpora (see 3A.3):

- Product corpus (from comparison_table.parquet): product existence,
  structured attributes, missing-information status. The authority for
  "does product X have attribute Y" and exhaustive "which products..."
  questions.
- Evidence corpus (from evidence.json): the retrieval documents used
  for lexical/semantic search and citations.

Reads comparison_table.parquet rather than the CSV: Step 2 writes list-
valued columns (modes/included_items/claims/specifications/variants) as
JSON strings in both files, but leaves scalar columns (price, bool
attributes, mode_count) as native types only in the Parquet -- avoiding
re-guessing "true"/"" string parsing that the CSV would require.
"""
from __future__ import annotations

import json
from typing import Any

import pyarrow.parquet as pq

from .config import COMPARISON_TABLE_PARQUET, EVIDENCE_JSON
from .models import ProductRecord, RetrievalDocument

IDENTITY_COLUMNS = {
    "product_id", "brand", "product_name", "product_url",
    "discovered_url", "source_listing_url", "retrieved_at",
}
JSON_ENCODED_COLUMNS = {
    "modes", "included_items", "claims", "specifications",
    "colors", "mode_details", "intensity_levels", "disclaimers", "claims_with_disclaimers",
    "pressure_sensing_details", "app_connectivity_details", "bluetooth_details", "battery_details", "charging_details", "coverage_tracking_details", "intensity_details",
}


def load_product_corpus(path=COMPARISON_TABLE_PARQUET) -> list[ProductRecord]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    products: list[ProductRecord] = []
    for row in rows:
        attributes: dict[str, Any] = {}
        for col, value in row.items():
            if col in IDENTITY_COLUMNS or col == "variants":
                continue
            if col in JSON_ENCODED_COLUMNS and isinstance(value, str):
                value = json.loads(value)
            attributes[col] = value
        variants = row.get("variants")
        if isinstance(variants, str):
            variants = json.loads(variants)
        products.append(
            ProductRecord(
                product_id=row["product_id"],
                brand=row["brand"],
                product_name=row["product_name"],
                product_url=row["product_url"],
                discovered_url=row["discovered_url"],
                source_listing_url=row["source_listing_url"],
                retrieved_at=row["retrieved_at"],
                attributes=attributes,
                variants=variants or [],
            )
        )
    products.sort(key=lambda p: p.product_id)
    return products


def load_evidence_records(path=EVIDENCE_JSON) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval_text(brand: str, product_name: str, attribute: str, evidence_text: str, section: str | None) -> str:
    """The contextual text that gets embedded/indexed -- per the 3A.5
    spec's example shape. evidence_text itself (the literal Step 2
    verbatim snippet) is kept separately on the document and is what
    citations ultimately point to."""
    return (
        f"Brand: {brand}\n"
        f"Product: {product_name}\n"
        f"Attribute: {attribute}\n"
        f"Evidence: {evidence_text}\n"
        f"Section: {section or 'N/A'}"
    )


def build_retrieval_documents(
    evidence_records: list[dict] | None = None,
    product_corpus: list[ProductRecord] | None = None,
) -> list[RetrievalDocument]:
    if evidence_records is None:
        evidence_records = load_evidence_records()
    if product_corpus is None:
        product_corpus = load_product_corpus()
    products_by_id = {p.product_id: p for p in product_corpus}

    documents: list[RetrievalDocument] = []
    for ev in evidence_records:
        product = products_by_id.get(ev["product_id"])
        brand = product.brand if product else ""
        product_name = product.product_name if product else ""
        documents.append(
            RetrievalDocument(
                document_id=f"doc:{ev['id']}",
                evidence_id=ev["id"],
                product_id=ev["product_id"],
                brand=brand,
                product_name=product_name,
                attribute=ev["attribute"],
                value=ev["value"],
                status=ev["status"],
                evidence_text=ev["evidence_text"],
                source_url=ev["source_url"],
                source_section=ev.get("source_section"),
                source_file=ev["source_file"],
                retrieval_text=_retrieval_text(
                    brand, product_name, ev["attribute"], ev["evidence_text"], ev.get("source_section")
                ),
            )
        )
    documents.sort(key=lambda d: d.document_id)
    return documents
