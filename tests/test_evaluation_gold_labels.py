"""Tests for the independent gold-label reader (3C.1). Confirms it
reads the real Step 2 files directly (pyarrow + json), never through
RetrievalEngine/src.rag.corpus."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from src.evaluation import gold_labels


class TestIndependenceFromRagObjects:
    def test_module_does_not_import_retrieval_engine_or_corpus(self):
        """Checks actual import statements (via AST), not prose in the
        module's own docstring explaining this independence rule."""
        tree = ast.parse(inspect.getsource(gold_labels))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        assert not any(m.startswith("src.rag") for m in imported_modules)


class TestLoadGoldCorpus:
    def test_loads_real_21_product_corpus(self):
        corpus = gold_labels.load_gold_corpus()
        assert len(corpus.products) == 21
        assert len(corpus.evidence) == len(json.loads((Path(__file__).resolve().parents[1] / "data" / "processed" / "evidence.json").read_text(encoding="utf-8")))

    def test_products_by_id_lookup(self):
        corpus = gold_labels.load_gold_corpus()
        product = corpus.products[0]
        assert corpus.products_by_id[product.product_id] is product

    def test_find_product_returns_none_for_unknown_id(self):
        corpus = gold_labels.load_gold_corpus()
        assert corpus.find_product("not-a-real-product") is None


class TestResolveStatus:
    def test_not_mentioned_when_no_evidence_exists(self):
        corpus = gold_labels.load_gold_corpus()
        product = next(p for p in corpus.products if p.get("charge_time_hours") is None
                        and not corpus.evidence_for(p.product_id, "charge_time_hours"))
        status, value, docs = corpus.resolve_status(product.product_id, "charge_time_hours")
        assert status == "not_mentioned"
        assert value is None
        assert docs == []

    def test_stated_with_real_value(self):
        corpus = gold_labels.load_gold_corpus()
        product = next(p for p in corpus.products if p.get("price") is not None)
        status, value, docs = corpus.resolve_status(product.product_id, "price")
        assert status == "stated"
        assert value == product.get("price")
        assert docs

    def test_stated_with_null_value_real_example(self):
        """battery_life_days has real stated+null examples in this corpus
        (confirmed by direct inspection) -- resolve_status must surface
        status=stated, value=None, not collapse it into not_mentioned."""
        corpus = gold_labels.load_gold_corpus()
        candidate = next(
            (p for p in corpus.products
             if p.get("battery_life_days") is None and corpus.evidence_for(p.product_id, "battery_life_days")),
            None,
        )
        assert candidate is not None, "expected a real stated+null battery_life_days example in the corpus"
        status, value, docs = corpus.resolve_status(candidate.product_id, "battery_life_days")
        assert status == "stated"
        assert value is None
        assert docs


class TestProductsWhere:
    def test_boolean_true_filter(self):
        corpus = gold_labels.load_gold_corpus()
        matches = corpus.products_where("pressure_sensing", True)
        assert len(matches) == 19

    def test_numeric_equality_filter(self):
        corpus = gold_labels.load_gold_corpus()
        matches = corpus.products_where("mode_count", 7)
        assert len(matches) == 3


class TestStatusCoverage:
    def test_confirms_known_coverage_gaps(self):
        coverage = gold_labels.status_coverage()
        assert coverage["stated_absent"]["has_real_example"] is False
        assert coverage["ambiguous"]["has_real_example"] is False

    def test_confirms_known_real_examples(self):
        coverage = gold_labels.status_coverage()
        assert coverage["not_mentioned"]["has_real_example"] is True
        assert coverage["stated_with_null_value"]["has_real_example"] is True
        # charging_method is "stated" with a null value where the page only says
        # "charger" (no magnetic wording): 16 product rows.
        assert coverage["stated_with_null_value"]["count"] == 13
