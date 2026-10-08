"""Tests for the test_cases.json schema validation (3C.1)."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.evaluation.schema import TestCase, TestSuite, load_test_suite

REAL_TEST_CASES_PATH = "data/evaluation/test_cases.json"


class TestLoadRealTestSuite:
    def test_loads_and_validates_the_real_file(self):
        suite = load_test_suite(REAL_TEST_CASES_PATH)
        assert len(suite.cases) >= 15

    def test_all_ids_are_unique(self):
        suite = load_test_suite(REAL_TEST_CASES_PATH)
        ids = [c.id for c in suite.cases]
        assert len(ids) == len(set(ids))

    def test_covers_required_categories(self):
        suite = load_test_suite(REAL_TEST_CASES_PATH)
        categories = {c.category for c in suite.cases}
        required = {
            "direct_attribute", "comparison", "exhaustive", "ambiguous_product",
            "out_of_corpus", "out_of_scope_source", "medical_advice",
            "prompt_injection_user", "prompt_injection_evidence",
        }
        assert required <= categories


class TestTestCaseValidation:
    def test_rejects_empty_question(self):
        with pytest.raises(ValidationError):
            TestCase(id="x", question="   ", category="direct_attribute",
                     expected_behavior="y", expected_answer_type="fact")

    def test_rejects_empty_id(self):
        with pytest.raises(ValidationError):
            TestCase(id="", question="q", category="direct_attribute",
                     expected_behavior="y", expected_answer_type="fact")

    def test_rejects_unknown_category(self):
        with pytest.raises(ValidationError):
            TestCase(id="x", question="q", category="not_a_real_category",
                     expected_behavior="y", expected_answer_type="fact")

    def test_defaults_are_sensible(self):
        case = TestCase(id="x", question="q", category="direct_attribute",
                         expected_behavior="y", expected_answer_type="fact")
        assert case.must_cite is True
        assert case.llm_expected is True
        assert case.gold_products == []
        assert case.exhaustive_condition_value is True


class TestTestSuiteValidation:
    def test_rejects_duplicate_ids(self):
        case_kwargs = dict(question="q", category="direct_attribute", expected_behavior="y", expected_answer_type="fact")
        with pytest.raises(ValidationError):
            TestSuite(version="v1", cases=[TestCase(id="dup", **case_kwargs), TestCase(id="dup", **case_kwargs)])

    def test_load_test_suite_rejects_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": "v1", "cases": [{"id": "x"}]}))
        with pytest.raises(ValidationError):
            load_test_suite(bad)
