"""Tests for the test_cases.json schema validation."""
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

    def test_covers_a_comparison_category(self):
        """"comparison" is the one category value that's semantically
        load-bearing (golden_dataset.py / application_eval.py both
        route on it for the comparison-arithmetic check) -- every other
        category is a free-form report label, so this is the only
        value worth asserting the suite actually contains."""
        suite = load_test_suite(REAL_TEST_CASES_PATH)
        categories = {c.category for c in suite.cases}
        assert "comparison" in categories

    def test_every_comparison_case_has_a_comparison_pair(self):
        suite = load_test_suite(REAL_TEST_CASES_PATH)
        for c in suite.cases:
            if c.category == "comparison":
                assert c.comparison_pair and len(c.comparison_pair) == 2, c.id


class TestTestCaseValidation:
    def test_rejects_empty_question(self):
        with pytest.raises(ValidationError):
            TestCase(id="x", question="   ", category="direct_attribute",
                     expected_behavior="y", expected_answer_type="fact")

    def test_rejects_empty_id(self):
        with pytest.raises(ValidationError):
            TestCase(id="", question="q", category="direct_attribute",
                     expected_behavior="y", expected_answer_type="fact")

    def test_category_is_free_form_not_a_closed_enum(self):
        """category/expected_answer_type are plain strings, not a
        Literal -- only the literal value "comparison" is ever
        pattern-matched by the harness (see schema.py's own docstring);
        every other value is just a report label, so a new case-type
        taxonomy never needs a schema change to be usable."""
        case = TestCase(id="x", question="q", category="a_brand_new_case_type",
                         expected_behavior="y", expected_answer_type="something_new")
        assert case.category == "a_brand_new_case_type"

    def test_defaults_are_sensible(self):
        case = TestCase(id="x", question="q", category="direct_attribute",
                         expected_behavior="y", expected_answer_type="fact")
        assert case.must_cite is True
        assert case.llm_expected is True
        assert case.gold_products == []
        assert case.must_include == []
        assert case.comparison_pair is None


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
