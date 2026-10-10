"""Pydantic schema for data/evaluation/test_cases.json.

Trimmed to exactly what the 4-layer DeepEval harness uses: `category` is a
plain report label now (nothing dispatches on it except the literal string
"comparison" -- golden_dataset.py and application_eval.py both check for
that one value to route to the comparison-arithmetic logic; every other
category value is never pattern-matched anywhere, only threaded through for
reporting). `expected_answer_type` is likewise never read by any code here
-- pure documentation. Both are free-form strings, not a closed enum, so
new case-type taxonomies (e.g. "false_premise", "conflicting_source_data")
don't require a code change to use -- only "comparison" is semantically
load-bearing.

`comparison_pair` is needed by application_eval.py's one remaining
deterministic check (comparison-arithmetic) and `gold_products`/
`gold_attributes` are needed by golden_dataset.py to build the gold
context/answer/claims.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TestCase(BaseModel):
    __test__ = False  # not a pytest test class, just named TestCase

    id: str
    question: str
    category: str
    expected_behavior: str
    gold_products: list[str] = Field(default_factory=list)
    gold_attributes: list[str] = Field(default_factory=list)
    expected_answer_type: str
    must_cite: bool = True
    llm_expected: bool = True
    notes: Optional[str] = None
    # Comparison-only: the two products under comparison, in the order
    # named in the question (for independent gold-diff computation).
    comparison_pair: Optional[list[str]] = None
    # Content checks on the answer text (src/evaluation/content_checks.py): every phrase in
    # must_include must appear, none in must_not_include may. Figures match as whole tokens.
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("id must be non-empty")
        return v

    @field_validator("question")
    @classmethod
    def _question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must be non-empty")
        return v


class TestSuite(BaseModel):
    __test__ = False  # not a pytest test class, just named TestSuite

    version: str
    cases: list[TestCase]

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, v: list[TestCase]) -> list[TestCase]:
        ids = [c.id for c in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate test case ids: {sorted(dupes)}")
        return v


def load_test_suite(path: Path) -> TestSuite:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TestSuite.model_validate(data)
