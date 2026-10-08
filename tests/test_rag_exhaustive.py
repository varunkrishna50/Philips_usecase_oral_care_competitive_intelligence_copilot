"""Tests for exhaustive "which products have X" queries (3A.13): a full
product-corpus scan, never top-k, with every true match included and
its own evidence attached."""
from __future__ import annotations

from src.rag import corpus
from src.rag.exhaustive import exhaustive_query
from src.rag.models import ProductRecord, RetrievalDocument


def _product(product_id, **attrs) -> ProductRecord:
    return ProductRecord(
        product_id=product_id, brand="Oral-B", product_name=product_id, product_url="x",
        discovered_url="x", source_listing_url="x", retrieved_at="2026-01-01T00:00:00Z",
        attributes=attrs,
    )


def _doc(product_id, attribute) -> RetrievalDocument:
    return RetrievalDocument(
        document_id=f"{product_id}:{attribute}", evidence_id=f"{product_id}:{attribute}",
        product_id=product_id, brand="Oral-B", product_name=product_id, attribute=attribute,
        value=True, status="stated", evidence_text="x", source_url="https://oralb.com/x",
        source_section="s", source_file="x.html", retrieval_text="x",
    )


class TestExhaustiveQuerySynthetic:
    def test_every_true_match_is_included(self):
        products = [
            _product("p1", pressure_sensing=True),
            _product("p2", pressure_sensing=False),
            _product("p3", pressure_sensing=None),
            _product("p4", pressure_sensing=True),
        ]
        docs = [_doc("p1", "pressure_sensing"), _doc("p4", "pressure_sensing")]
        matches = exhaustive_query("pressure_sensing", products, docs)
        assert {m.product.product_id for m in matches} == {"p1", "p4"}

    def test_each_match_carries_its_own_evidence_only(self):
        products = [_product("p1", timer=True), _product("p2", timer=True)]
        docs = [_doc("p1", "timer"), _doc("p2", "timer")]
        matches = exhaustive_query("timer", products, docs)
        by_id = {m.product.product_id: m.evidence for m in matches}
        assert [d.product_id for d in by_id["p1"]] == ["p1"]
        assert [d.product_id for d in by_id["p2"]] == ["p2"]

    def test_missing_evidence_does_not_exclude_a_true_match(self):
        """A product with value=True but (hypothetically) no retrieval
        document must still appear -- the product table, not evidence
        presence, decides inclusion."""
        products = [_product("p1", timer=True)]
        matches = exhaustive_query("timer", products, documents_but_no_evidence_for_p1 := [])
        assert len(matches) == 1
        assert matches[0].evidence == []

    def test_false_and_null_are_excluded(self):
        products = [_product("p1", timer=False), _product("p2", timer=None)]
        matches = exhaustive_query("timer", products, [])
        assert matches == []


class TestExhaustiveQueryRealData:
    def test_pressure_sensing_count_matches_product_corpus_exactly(self):
        products = corpus.load_product_corpus()
        documents = corpus.build_retrieval_documents(product_corpus=products)
        expected = sum(1 for p in products if p.attributes.get("pressure_sensing") is True)
        matches = exhaustive_query("pressure_sensing", products, documents)
        assert len(matches) == expected
        assert expected > 0  # sanity: the real corpus does have some

    def test_every_matched_product_has_at_least_one_piece_of_evidence(self):
        products = corpus.load_product_corpus()
        documents = corpus.build_retrieval_documents(product_corpus=products)
        matches = exhaustive_query("pressure_sensing", products, documents)
        for m in matches:
            assert len(m.evidence) >= 1, f"{m.product.product_id} has pressure_sensing=True but no evidence"
