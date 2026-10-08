"""Pydantic schema for data/evaluation/test_cases.json (3C.1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Category = Literal[
    "direct_attribute", "product_specific", "comparison", "exhaustive",
    "status_not_mentioned", "status_stated_absent", "status_ambiguous", "status_stated_null",
    "ambiguous_product", "out_of_corpus", "out_of_scope_source", "medical_advice",
    "prompt_injection_user", "prompt_injection_evidence",
]

ExpectedAnswerType = Literal[
    "fact", "comparison", "exhaustive_list", "not_mentioned_disclosure",
    "stated_absent_disclosure", "ambiguous_disclosure", "stated_null_disclosure",
    "clarification", "refusal_not_in_corpus", "refusal_out_of_scope", "refusal_or_claims_only",
    "no_fabrication",
]


class TestCase(BaseModel):
    __test__ = False  # not a pytest test class, just named TestCase per the 3C.1 spec

    id: str
    question: str
    category: Category
    expected_behavior: str
    gold_products: list[str] = Field(default_factory=list)
    gold_attributes: list[str] = Field(default_factory=list)
    expected_answer_type: ExpectedAnswerType
    must_cite: bool = True
    llm_expected: bool = True
    notes: Optional[str] = None
    # Comparison-only: the two products under comparison, in the order
    # named in the question (for independent gold-diff computation).
    comparison_pair: Optional[list[str]] = None
    # status_* categories: the attribute whose status is under test.
    status_attribute: Optional[str] = None
    # exhaustive category only: the value an attribute must equal for a
    # product to count as a match. Defaults to True (the boolean
    # attributes 3A.13 was designed for); set explicitly for a numeric
    # exhaustive condition like "7 brushing modes".
    exhaustive_condition_value: Any = True
    # poisoned-evidence injection only: which real evidence_id to clone
    # and append instruction-like text to, in memory, for --live only.
    poison_base_evidence_id: Optional[str] = None

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
    __test__ = False  # not a pytest test class, just named TestSuite per the 3C.1 spec

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
