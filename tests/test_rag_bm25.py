"""Tests for BM25 lexical retrieval over synthetic documents (fast, no
embedding model). Covers exact product names/IDs/SKUs/attribute terms
per 3A.4.
"""
from __future__ import annotations

from src.rag.bm25_index import BM25Index
from src.rag.models import RetrievalDocument


def _doc(document_id, product_id, attribute, evidence_text, retrieval_text=None) -> RetrievalDocument:
    return RetrievalDocument(
        document_id=document_id, evidence_id=document_id, product_id=product_id,
        brand="Oral-B", product_name=product_id, attribute=attribute, value=True,
        status="stated", evidence_text=evidence_text, source_url="https://oralb.com/x",
        source_section="Features", source_file="x.html",
        retrieval_text=retrieval_text or f"Product: {product_id}\nAttribute: {attribute}\nEvidence: {evidence_text}",
    )


DOCS = [
    _doc("d1", "oralb_io-series-9", "pressure_sensing", "Smart Pressure Sensor"),
    _doc("d2", "oralb_io-series-10", "mode_count", "7 Smart Brushing Modes"),
    _doc("d3", "philips_hx9990-13", "price", "SKU HX9990/13 priced at $379.96"),
]


class TestBM25Search:
    def test_exact_attribute_term_retrieves_the_right_document(self):
        index = BM25Index(DOCS)
        results = index.search("pressure sensor", top_k=5)
        assert results
        assert results[0][0].document_id == "d1"

    def test_product_id_token_retrieves_the_right_document(self):
        index = BM25Index(DOCS)
        results = index.search("io-series-10 modes", top_k=5)
        assert results[0][0].document_id == "d2"

    def test_sku_token_retrieves_the_right_document(self):
        index = BM25Index(DOCS)
        results = index.search("HX9990/13 price", top_k=5)
        assert results
        assert results[0][0].document_id == "d3"

    def test_top_k_is_respected(self):
        index = BM25Index(DOCS)
        results = index.search("toothbrush pressure modes price", top_k=1)
        assert len(results) <= 1

    def test_empty_query_returns_nothing(self):
        index = BM25Index(DOCS)
        assert index.search("", top_k=5) == []

    def test_empty_corpus_returns_nothing(self):
        index = BM25Index([])
        assert index.search("anything", top_k=5) == []

    def test_no_matching_terms_returns_nothing(self):
        index = BM25Index(DOCS)
        assert index.search("zzz nonexistent qqq", top_k=5) == []
