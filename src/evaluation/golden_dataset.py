"""Golden dataset for the DeepEval-based evaluation: one GoldenCase per
data/evaluation/test_cases.json TestCase, built entirely from that
file's own hand-authored fields -- `expected_behavior` is the golden
answer, `must_include` doubles as the list of atomic claims a complete
answer must cover.

This is a deliberate trade-off, made explicitly at the user's request:
earlier this evaluated against an independently-loaded GoldCorpus
(parquet/evidence.json, read without touching any src.rag module), so a
bug shared between the pipeline and its own gold-label builder could
never cancel out. That independence is gone now -- test_cases.json is
hand-written and may itself encode the same mistake the pipeline makes
-- but the project owner wrote every case and attests its contents are
correct, and wants the eval suite to trust that single source rather
than maintain a second, auto-derived one that can drift from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.models import RetrievalDocument

from .schema import TestCase


@dataclass
class GoldenCase:
    id: str
    question: str
    category: str
    test_case: TestCase
    golden_answer: str = ""
    expected_claims: list[str] = field(default_factory=list)
    # [golden_answer] + expected_claims -- "known true" context for the
    # Groundedness check (pipeline_eval.py), which used to be raw gold
    # evidence_text. Both halves are hand-authored in test_cases.json now.
    golden_context: list[str] = field(default_factory=list)


def build_golden_case(case: TestCase) -> GoldenCase:
    golden_answer = case.expected_behavior
    expected_claims = list(case.must_include)
    golden_context = [golden_answer] + expected_claims
    return GoldenCase(
        id=case.id,
        question=case.question,
        category=case.category,
        test_case=case,
        golden_answer=golden_answer,
        expected_claims=expected_claims,
        golden_context=golden_context,
    )


def build_golden_dataset(cases: list[TestCase]) -> list[GoldenCase]:
    return [build_golden_case(c) for c in cases]


def golden_retrieval_documents(golden: GoldenCase, retrieval_documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
    """The REAL RetrievalDocument objects (the same ones production
    uses) whose (product_id, attribute) is named by this case's
    gold_products x gold_attributes -- selected by that membership, not
    by the retrieval ranking algorithm, so generator_eval.py can call
    the real generation hook on known-good context without reimplementing
    RetrievalDocument construction (same technique runner.py's own
    poisoned-evidence test already uses)."""
    case = golden.test_case
    if not case.gold_products or not case.gold_attributes:
        return []
    wanted_products = set(case.gold_products)
    wanted_attributes = set(case.gold_attributes)
    return [
        d for d in retrieval_documents
        if d.product_id in wanted_products and d.attribute in wanted_attributes
    ]
