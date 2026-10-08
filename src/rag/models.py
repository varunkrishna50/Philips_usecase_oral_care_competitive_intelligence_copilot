"""Lightweight data structures for the retrieval layer.

Deliberately NOT importing src.normalization's pydantic classes: this
package only ever reads Step 2's already-serialized output files
(comparison_table.csv, evidence.json), so it has its own independent
shapes rather than reaching back into Step 2's internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RetrievalDocument:
    """One evidence record, reshaped for retrieval. Every field traces
    straight back to a Step 2 evidence.json record -- see corpus.py's
    build_retrieval_documents(). The invariant this preserves:
    retrieval document -> evidence_id -> Step 2 evidence -> product +
    attribute + source URL. No retrieval result loses provenance.
    """

    document_id: str
    evidence_id: str
    product_id: str
    brand: str
    product_name: str
    attribute: str
    value: Any
    status: str
    evidence_text: str
    source_url: str
    source_section: Optional[str]
    source_file: str
    # The text actually embedded/indexed (contextual: brand/product/
    # attribute/evidence/section) -- kept separate from evidence_text,
    # which stays the literal verbatim Step 2 evidence.
    retrieval_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "evidence_id": self.evidence_id,
            "product_id": self.product_id,
            "brand": self.brand,
            "product_name": self.product_name,
            "attribute": self.attribute,
            "value": self.value,
            "status": self.status,
            "evidence_text": self.evidence_text,
            "source_url": self.source_url,
            "source_section": self.source_section,
            "source_file": self.source_file,
            "retrieval_text": self.retrieval_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalDocument":
        return cls(**data)


@dataclass
class ProductRecord:
    """One row of comparison_table.csv, with attribute values kept in a
    single dict (rather than 18 named fields) so this module doesn't
    have to track Step 2's ATTRIBUTE_FIELDS list by hand."""

    product_id: str
    brand: str
    product_name: str
    product_url: str
    discovered_url: str
    source_listing_url: str
    retrieved_at: str
    attributes: dict[str, Any] = field(default_factory=dict)
    variants: list[dict] = field(default_factory=list)

    def get(self, attribute: str) -> Any:
        return self.attributes.get(attribute)


@dataclass
class ScoredDocument:
    """A retrieval candidate carrying every intermediate score so
    --debug output and the evaluation harness can show exactly how a
    result was ranked."""

    document: RetrievalDocument
    bm25_score: float = 0.0
    bm25_norm: float = 0.0
    semantic_score: float = 0.0
    semantic_norm: float = 0.0
    hybrid_score: float = 0.0
    rerank_tier: int = 99
    rerank_score: float = 0.0
