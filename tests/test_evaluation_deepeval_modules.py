"""Unit tests for the DeepEval-based eval modules' own logic (golden
dataset construction, the offline judge-call/classifier gates, and the
one remaining deterministic metric) -- independent of the full offline
integration run covered by test_evaluation_runner.py.
"""
from __future__ import annotations

from pathlib import Path

from src.evaluation import application_eval, golden_dataset, retriever_eval
from src.evaluation.deepeval_support import MetricOutcome, classify_if_live, measure_if_live
from src.evaluation.gold_labels import load_gold_corpus
from src.evaluation.schema import TestCase, load_test_suite

ROOT = Path(__file__).resolve().parents[1]


def _case(**kw) -> TestCase:
    base = dict(id="c1", question="q", category="direct_attribute", expected_behavior="fallback text", expected_answer_type="fact")
    return TestCase(**{**base, **kw})


# --- golden_dataset ------------------------------------------------------------------------------------

def test_direct_attribute_case_builds_one_claim_per_product_attribute_pair():
    corpus = load_gold_corpus()
    product = next(p for p in corpus.products if isinstance(p.get("price"), (int, float)) and not isinstance(p.get("price"), bool))
    case = _case(gold_products=[product.product_id], gold_attributes=["price"])
    claims = golden_dataset.build_expected_claims(corpus, case)
    assert len(claims) == 1
    assert product.product_name in claims[0] and str(product.get("price")) in claims[0]


def test_comparison_claims_describe_both_products_for_a_boolean_attribute():
    corpus = load_gold_corpus()
    a, b = corpus.products[0], corpus.products[1]
    case = _case(category="comparison", comparison_pair=[a.product_id, b.product_id],
                 gold_attributes=["pressure_sensing"], gold_products=[a.product_id, b.product_id],
                 expected_answer_type="comparison")
    claims = golden_dataset.build_expected_claims(corpus, case)
    assert any(a.product_name in c for c in claims)
    assert any(b.product_name in c for c in claims)


def test_golden_answer_falls_back_to_expected_behavior_when_nothing_else_is_available():
    corpus = load_gold_corpus()
    case = _case(category="out_of_corpus", gold_products=[], expected_answer_type="refusal_not_in_corpus")
    golden = golden_dataset.build_golden_case(case, corpus)
    assert golden.golden_answer == "fallback text"
    assert golden.golden_context == [] and golden.expected_claims == []


def test_build_golden_dataset_covers_every_real_test_case():
    suite = load_test_suite(ROOT / "data" / "evaluation" / "test_cases.json")
    corpus = load_gold_corpus()
    goldens = golden_dataset.build_golden_dataset(suite.cases, corpus)
    assert len(goldens) == len(suite.cases) >= 15
    assert {g.id for g in goldens} == {c.id for c in suite.cases}


# --- deepeval_support.measure_if_live / classify_if_live: offline never constructs anything ------------------

def test_offline_mode_never_calls_the_metric_factory():
    def _boom():
        raise AssertionError("offline mode must never construct a DeepEval metric (it would need a real API key)")

    outcome = measure_if_live("SomeMetric", _boom, test_case=None, mode="offline")
    assert outcome == MetricOutcome("SomeMetric", exercised=False, error="offline mode: DeepEval judge calls are disabled")


def test_live_mode_degrades_to_not_exercised_if_construction_fails():
    def _raises():
        raise RuntimeError("no OPENAI_API_KEY configured")

    outcome = measure_if_live("SomeMetric", _raises, test_case=None, mode="live")
    assert not outcome.exercised and "no OPENAI_API_KEY" in outcome.error


def test_live_mode_calls_measure_on_the_constructed_metric(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # judge_available() just needs a key present, never dials out here
    calls = []

    class FakeMetric:
        def measure(self, test_case, _show_indicator=True, _in_component=False):
            calls.append(test_case)
            self.score, self.success, self.reason = 0.9, True, "looks fine"

        def is_successful(self):
            return self.success

    outcome = measure_if_live("FakeMetric", FakeMetric, test_case="a-test-case", mode="live")
    assert calls == ["a-test-case"]
    assert outcome.exercised and outcome.score == 0.9 and outcome.passed is True


def test_offline_mode_never_calls_the_classifier_factory():
    def _boom():
        raise AssertionError("offline mode must never construct a DeepEval classifier")

    outcome = classify_if_live("SomeClassifier", _boom, "resisted", test_case=None, mode="offline")
    assert not outcome.exercised and outcome.error == "offline mode: DeepEval judge calls are disabled"


def test_live_mode_calls_classify_and_checks_the_expected_label(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeClassifier:
        def classify(self, test_case, _show_indicator=True, _in_component=False):
            self.label, self.reason = "resisted", "ignored the embedded instruction"

        def is_successful(self, expected_label):
            return self.label == expected_label

    outcome = classify_if_live("FakeClassifier", FakeClassifier, "resisted", test_case="tc", mode="live")
    assert outcome.exercised and outcome.passed is True and "resisted" in outcome.reason

    class FakeFailingClassifier(FakeClassifier):
        def classify(self, test_case, _show_indicator=True, _in_component=False):
            self.label, self.reason = "followed_injection", "complied with the override"

    outcome = classify_if_live("FakeClassifier", FakeFailingClassifier, "resisted", test_case="tc", mode="live")
    assert outcome.exercised and outcome.passed is False


# --- retriever_eval: document scope ---------------------------------------------------------------------------

def _golden(category, gold_products):
    case = _case(category=category, gold_products=gold_products)
    return golden_dataset.GoldenCase(id="c1", question="q", category=category, test_case=case)


def test_has_document_scope_true_only_for_cases_with_gold_products():
    assert retriever_eval.has_document_scope(_golden("direct_attribute", ["p1"]))
    assert not retriever_eval.has_document_scope(_golden("out_of_corpus", []))


def test_chunks_from_retrieval_result_prefixes_each_chunk_with_its_product_name():
    """Without the product-name prefix, a comparison case's two bare
    price chunks ("$199.96", "$329.99") give a relevancy judge no way
    to tell which product either one belongs to."""
    from types import SimpleNamespace

    doc = SimpleNamespace(product_name="Oral-B iO Series 9", shown_text="$329.99")
    result = SimpleNamespace(exhaustive_matches=None, comparison_results=None,
                              results=[SimpleNamespace(document=doc)])
    assert retriever_eval.chunks_from_retrieval_result(result) == ["Oral-B iO Series 9: $329.99"]


def test_contextual_relevancy_is_skipped_for_by_design_empty_retrieval_categories(real_engine):
    """medical_advice and missing_attribute cases retrieve correctly
    even when nothing in the corpus answers the literal question --
    ContextualRelevancyMetric can't tell "correctly found nothing" from
    a real miss, so it's never even constructed for these categories."""
    corpus = load_gold_corpus()
    product = corpus.products[0]
    question = f"What is the price of the {product.product_name}?"

    excluded = _case(category="missing_attribute", question=question, gold_products=[product.product_id])
    excluded_golden = golden_dataset.GoldenCase(id="c1", question=question, category="missing_attribute", test_case=excluded)
    excluded_result = retriever_eval.evaluate_case(excluded_golden, real_engine, mode="offline")
    assert "ContextualRelevancyMetric" not in {m.name for m in excluded_result.metrics}
    assert "ContextualPrecisionMetric" in {m.name for m in excluded_result.metrics}

    ordinary = _case(category="direct_attribute", question=question, gold_products=[product.product_id])
    ordinary_golden = golden_dataset.GoldenCase(id="c2", question=question, category="direct_attribute", test_case=ordinary)
    ordinary_result = retriever_eval.evaluate_case(ordinary_golden, real_engine, mode="offline")
    assert "ContextualRelevancyMetric" in {m.name for m in ordinary_result.metrics}


# --- application_eval: comparison arithmetic (the one remaining deterministic metric) -------------------------

def test_comparison_arithmetic_passes_for_a_real_comparison_case(real_engine):
    corpus = load_gold_corpus()
    suite = load_test_suite(ROOT / "data" / "evaluation" / "test_cases.json")
    case = next(c for c in suite.cases if c.category == "comparison")
    golden = golden_dataset.build_golden_case(case, corpus)

    outcome = application_eval._comparison_arithmetic(golden, real_engine, corpus)
    assert outcome.exercised and outcome.passed is True and outcome.score == 1.0


def test_comparison_arithmetic_fails_when_the_pipeline_disagrees_with_gold(real_engine, monkeypatch):
    corpus = load_gold_corpus()
    suite = load_test_suite(ROOT / "data" / "evaluation" / "test_cases.json")
    # Numeric or boolean, whichever the suite's first comparison case happens to use --
    # _wrong_gold below corrupts either kind, so this doesn't depend on attribute type.
    case = next(c for c in suite.cases if c.category == "comparison")
    golden = golden_dataset.build_golden_case(case, corpus)

    from dataclasses import replace

    from src.evaluation import comparison_eval as comparison_eval_mod

    real_compute = comparison_eval_mod.compute_gold_comparison

    def _wrong_gold(corpus_arg, product_a_id, product_b_id, attribute):
        gold = real_compute(corpus_arg, product_a_id, product_b_id, attribute)
        if gold.diff is not None:
            return replace(gold, diff=gold.diff + 9999)
        if gold.a_true is not None:
            return replace(gold, a_true=not gold.a_true)
        return gold

    monkeypatch.setattr(comparison_eval_mod, "compute_gold_comparison", _wrong_gold)
    outcome = application_eval._comparison_arithmetic(golden, real_engine, corpus)
    assert outcome.exercised and outcome.passed is False
