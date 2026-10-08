"""Tests for src.rag.corpus: the product corpus / evidence corpus split
(3A.3) and the provenance invariant (3A.2): retrieval document ->
evidence_id -> Step 2 evidence -> product + attribute + source URL.
"""
from __future__ import annotations

import json

from src.rag import corpus
from src.rag.config import EVIDENCE_JSON


def test_product_corpus_matches_actual_step2_output():
    products = corpus.load_product_corpus()
    assert len(products) == 24
    brands = {p.brand for p in products}
    assert brands == {"Philips", "Oral-B"}
    assert sum(1 for p in products if p.brand == "Oral-B") == 13
    assert sum(1 for p in products if p.brand == "Philips") == 11


def test_product_corpus_ids_are_unique():
    products = corpus.load_product_corpus()
    ids = [p.product_id for p in products]
    assert len(ids) == len(set(ids))


def test_product_corpus_list_attributes_are_decoded_from_json():
    products = corpus.load_product_corpus()
    for p in products:
        for attr in ("modes", "included_items", "claims", "specifications"):
            value = p.attributes.get(attr)
            assert value is None or isinstance(value, list), f"{p.product_id}.{attr} should be list or None, got {type(value)}"


def test_retrieval_document_count_matches_evidence_json():
    evidence_records = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    documents = corpus.build_retrieval_documents()
    assert len(documents) == len(evidence_records)


def test_every_retrieval_document_traces_back_to_its_evidence_record():
    """The 3A.2 provenance invariant, checked mechanically: every
    document's evidence_id resolves to a real Step 2 evidence record
    whose product_id/attribute/source_url match the document exactly."""
    evidence_by_id = {e["id"]: e for e in json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))}
    documents = corpus.build_retrieval_documents()
    assert documents  # non-empty corpus
    for doc in documents:
        ev = evidence_by_id.get(doc.evidence_id)
        assert ev is not None, f"{doc.document_id} references missing evidence_id {doc.evidence_id}"
        assert ev["product_id"] == doc.product_id
        assert ev["attribute"] == doc.attribute
        assert ev["source_url"] == doc.source_url
        assert ev["evidence_text"] == doc.evidence_text
        assert ev["status"] == doc.status


def test_retrieval_text_embeds_brand_product_attribute_evidence_section():
    documents = corpus.build_retrieval_documents()
    doc = next(d for d in documents if d.brand and d.source_section)
    assert f"Brand: {doc.brand}" in doc.retrieval_text
    assert f"Product: {doc.product_name}" in doc.retrieval_text
    assert f"Attribute: {doc.attribute}" in doc.retrieval_text
    assert doc.evidence_text in doc.retrieval_text
    assert doc.source_section in doc.retrieval_text


def test_product_corpus_and_evidence_corpus_are_independent_layers():
    """Absence of a retrieval document for (product, attribute) must
    never be used to conclude not_mentioned -- that status comes only
    from the product corpus. Pick any product+attribute pair with zero
    evidence documents and confirm the product corpus still reports a
    real status-bearing value (None for not_mentioned, not an error)."""
    products = corpus.load_product_corpus()
    documents = corpus.build_retrieval_documents(product_corpus=products)
    covered = {(d.product_id, d.attribute) for d in documents}

    # price_raw is a flattened passthrough of price.raw_value (see
    # src/normalization/models.py:to_flat_dict) -- it has no evidence
    # record under its own name (that evidence is filed as attribute
    # "price"), so it's excluded from this per-attribute invariant.
    found_uncovered = False
    for p in products:
        for attr, value in p.attributes.items():
            if attr == "price_raw":
                continue
            if (p.product_id, attr) not in covered:
                found_uncovered = True
                assert value is None  # not_mentioned in the product corpus
    assert found_uncovered, "expected at least one not_mentioned product+attribute pair in real data"
