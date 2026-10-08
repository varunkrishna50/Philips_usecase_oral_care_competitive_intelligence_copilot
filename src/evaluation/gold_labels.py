"""Independent gold-label reader for the 3C.1 evaluation harness.

Reads `comparison_table.parquet` and `evidence.json` directly (pyarrow +
json only) -- this module deliberately does NOT import
`src.rag.corpus`, `src.rag.retrieve`, or any other RAG component under
test, so that a bug shared between the system and its own gold-label
builder can never cancel out (3C.1 integrity rule). pandas is not a
project dependency (not installed in this environment); pyarrow already
is (Step 2), so it's used here instead -- same source files, same
result, no new dependency.

Status model (mirrors src/normalization/models.py's four statuses, but
computed independently here):
- "stated": an evidence record exists for (product, attribute) with a
  non-null comparison_table value.
- "stated" with value=None: an evidence record exists but the
  comparison_table value is null (discussed but unnormalizable).
- "not_mentioned": no evidence record exists for (product, attribute).
- "stated_absent" / "ambiguous": status values the Step 2 schema
  supports, but (confirmed by direct inspection) zero of the 486 real
  evidence records in this corpus carry them -- see status_coverage().
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
COMPARISON_TABLE_PARQUET = ROOT / "data" / "processed" / "comparison_table.parquet"
EVIDENCE_JSON = ROOT / "data" / "processed" / "evidence.json"

JSON_ENCODED_COLUMNS = {"modes", "included_items", "claims", "specifications"}

ALL_STATUSES = ("stated", "not_mentioned", "stated_absent", "ambiguous")


@dataclass
class GoldProduct:
    product_id: str
    brand: str
    product_name: str
    product_url: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def get(self, attribute: str) -> Any:
        return self.attributes.get(attribute)


@dataclass
class GoldEvidence:
    evidence_id: str
    product_id: str
    attribute: str
    value: Any
    status: str
    evidence_text: str
    source_url: str


def load_gold_products(path: Path = COMPARISON_TABLE_PARQUET) -> list[GoldProduct]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    products = []
    for row in rows:
        attributes: dict[str, Any] = {}
        for col, value in row.items():
            if col in ("product_id", "brand", "product_name", "product_url", "discovered_url",
                        "source_listing_url", "retrieved_at", "variants"):
                continue
            if col in JSON_ENCODED_COLUMNS and isinstance(value, str):
                value = json.loads(value)
            attributes[col] = value
        products.append(GoldProduct(
            product_id=row["product_id"], brand=row["brand"], product_name=row["product_name"],
            product_url=row["product_url"], attributes=attributes,
        ))
    products.sort(key=lambda p: p.product_id)
    return products


def load_gold_evidence(path: Path = EVIDENCE_JSON) -> list[GoldEvidence]:
    records = json.loads(path.read_text(encoding="utf-8"))
    evidence = [
        GoldEvidence(
            evidence_id=r["id"], product_id=r["product_id"], attribute=r["attribute"],
            value=r["value"], status=r["status"], evidence_text=r["evidence_text"],
            source_url=r["source_url"],
        )
        for r in records
    ]
    evidence.sort(key=lambda e: e.evidence_id)
    return evidence


@dataclass
class GoldCorpus:
    products: list[GoldProduct]
    evidence: list[GoldEvidence]

    @property
    def products_by_id(self) -> dict[str, GoldProduct]:
        return {p.product_id: p for p in self.products}

    def evidence_for(self, product_id: str, attribute: str) -> list[GoldEvidence]:
        return [e for e in self.evidence if e.product_id == product_id and e.attribute == attribute]

    def resolve_status(self, product_id: str, attribute: str) -> tuple[str, Any, list[GoldEvidence]]:
        """Independently reproduces the four-status resolution: status
        comes from whether an evidence record exists, never from
        whether retrieval happened to surface one."""
        docs = self.evidence_for(product_id, attribute)
        if not docs:
            return "not_mentioned", None, []
        status = docs[0].status
        value = self.products_by_id[product_id].get(attribute)
        return status, value, docs

    def products_where(self, attribute: str, value: Any = True) -> list[GoldProduct]:
        return [p for p in self.products if p.get(attribute) == value]

    def find_product(self, product_id: str) -> Optional[GoldProduct]:
        return self.products_by_id.get(product_id)


def load_gold_corpus() -> GoldCorpus:
    return GoldCorpus(products=load_gold_products(), evidence=load_gold_evidence())


def status_coverage(corpus: Optional[GoldCorpus] = None) -> dict[str, dict]:
    """Which of the four Step 2 statuses -- plus the distinguished
    "stated with a null normalized value" sub-case the spec calls out
    separately -- have at least one real occurrence in this corpus, and
    for which (product, attribute) pairs. Used both to pick real test
    cases and to report honest coverage gaps (3C.1 integrity rule:
    never fabricate a missing status)."""
    corpus = corpus or load_gold_corpus()
    by_status: dict[str, list[dict]] = {s: [] for s in ALL_STATUSES}
    stated_with_null_value: list[dict] = []

    evidence_keys = {(e.product_id, e.attribute) for e in corpus.evidence}
    attributes = sorted({e.attribute for e in corpus.evidence})
    products_by_id = corpus.products_by_id

    for e in corpus.evidence:
        if e.status in by_status:
            by_status[e.status].append({"product_id": e.product_id, "attribute": e.attribute})
        if e.status == "stated" and products_by_id[e.product_id].get(e.attribute) is None:
            stated_with_null_value.append({"product_id": e.product_id, "attribute": e.attribute})

    # not_mentioned: every (product, attribute) pair with no evidence at
    # all, restricted to attributes that appear SOMEWHERE in the corpus
    # (so we don't manufacture "coverage" for nonsense attribute names).
    for product in corpus.products:
        for attribute in attributes:
            if (product.product_id, attribute) not in evidence_keys:
                by_status["not_mentioned"].append({"product_id": product.product_id, "attribute": attribute})

    result = {
        status: {"count": len(examples), "examples": examples[:5], "has_real_example": len(examples) > 0}
        for status, examples in by_status.items()
    }
    result["stated_with_null_value"] = {
        "count": len(stated_with_null_value),
        "examples": stated_with_null_value[:5],
        "has_real_example": len(stated_with_null_value) > 0,
    }
    return result
