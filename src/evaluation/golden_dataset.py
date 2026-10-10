"""Golden dataset for the DeepEval-based evaluation: one GoldenCase per
data/evaluation/test_cases.json TestCase, built entirely from the
independently-loaded GoldCorpus (gold_labels.py) -- never from
RetrievalEngine's own ranking, same independence rule gold_labels.py
already states for the rest of this package.

`golden_answer` and `expected_claims` are MECHANICAL references
assembled from gold evidence/attribute values, not human-written prose.
That's a real limitation, stated plainly rather than hidden (same habit
as groundedness.py's own docstring): they're good enough for DeepEval's
semantic (meaning-level) metrics to judge against, not literary gold
answers. `golden_context` is the raw `evidence_text` GoldCorpus holds
(it deliberately has no RetrievalEngine-rendered `shown_text` -- that
would reintroduce the dependency this module exists to avoid).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.rag.models import RetrievalDocument

from . import comparison_eval
from .gold_labels import GoldCorpus, GoldEvidence
from .schema import TestCase


def attribute_label(attribute: str) -> str:
    return attribute.replace("_", " ")


@dataclass
class GoldenCase:
    id: str
    question: str
    category: str
    test_case: TestCase
    golden_context: list[str] = field(default_factory=list)
    golden_evidence_ids: list[str] = field(default_factory=list)
    golden_answer: str = ""
    expected_claims: list[str] = field(default_factory=list)


def _gold_evidence_for_case(corpus: GoldCorpus, case: TestCase) -> list[GoldEvidence]:
    if not case.gold_attributes or not case.gold_products:
        return []
    out: list[GoldEvidence] = []
    for product_id in case.gold_products:
        for attribute in case.gold_attributes:
            out.extend(corpus.evidence_for(product_id, attribute))
    return out


def _attribute_claim(corpus: GoldCorpus, product_id: str, attribute: str) -> Optional[str]:
    product = corpus.find_product(product_id)
    if product is None:
        return None
    status, value, docs = corpus.resolve_status(product_id, attribute)
    label = attribute_label(attribute)
    if status == "not_mentioned":
        return f"{product.product_name}: the page does not mention {label}."
    if status == "stated_absent":
        return f"{product.product_name}: the page states it does not have {label}."
    if status == "ambiguous":
        return f"{product.product_name}: the page gives conflicting information about {label}."
    if value is None:
        text = docs[0].evidence_text if docs else ""
        return f'{product.product_name}: the page states "{text}" about {label}.'
    if isinstance(value, bool):
        return f"{product.product_name} {'has' if value else 'does not have'} {label}."
    return f"{product.product_name} {label}: {value}."


def _comparison_claims(corpus: GoldCorpus, case: TestCase) -> list[str]:
    product_a_id, product_b_id = case.comparison_pair
    product_a, product_b = corpus.find_product(product_a_id), corpus.find_product(product_b_id)
    claims: list[str] = []
    for attribute in case.gold_attributes:
        gold = comparison_eval.compute_gold_comparison(corpus, product_a_id, product_b_id, attribute)
        label = attribute_label(attribute)
        if not gold.comparable:
            claims.append(f"{product_a.product_name} and {product_b.product_name} are not safely comparable on {label}.")
        elif gold.diff is not None:
            # Per-product values first (mirrors the boolean branch below) -- a completeness/correctness
            # judge needs the actual per-product numbers to match against a real answer's table cells,
            # not just the derived gap, which isn't necessarily stated as its own explicit sentence.
            claims.append(f"{product_a.product_name} {label}: {gold.value_a}.")
            claims.append(f"{product_b.product_name} {label}: {gold.value_b}.")
            # round(..., 10) just for display -- clears float noise like 79.96-159.96 -> 80.00000000000001;
            # the real comparison-arithmetic check (application_eval.py) still compares against gold.diff directly
            claims.append(f"{product_a.product_name} and {product_b.product_name} differ by {round(gold.diff, 10)} on {label}.")
        else:
            claims.append(f"{product_a.product_name} {'has' if gold.a_true else 'does not have'} {label}.")
            claims.append(f"{product_b.product_name} {'has' if gold.b_true else 'does not have'} {label}.")
    return claims


def build_expected_claims(corpus: GoldCorpus, case: TestCase) -> list[str]:
    """Atomic, independently-sourced claim strings -- one per gold
    (product, attribute) pair -- used by application_eval.py's
    Completeness metric. Never built from the pipeline's own output."""
    if case.category == "comparison" and case.comparison_pair:
        return _comparison_claims(corpus, case)
    claims = []
    for product_id in case.gold_products:
        for attribute in case.gold_attributes:
            claim = _attribute_claim(corpus, product_id, attribute)
            if claim:
                claims.append(claim)
    return claims


def build_golden_answer(corpus: GoldCorpus, case: TestCase, evidence: list[GoldEvidence], claims: list[str]) -> str:
    if claims:
        return " ".join(claims)
    if evidence:
        products_by_id = corpus.products_by_id
        return " ".join(f"{products_by_id[e.product_id].product_name}: {e.evidence_text}" for e in evidence)
    # Safety/disposition categories have no gold products/attributes to
    # build claims from -- expected_behavior (already hand-written per
    # case in test_cases.json) is the closest thing to a reference.
    return case.expected_behavior


def build_golden_case(case: TestCase, corpus: GoldCorpus) -> GoldenCase:
    evidence = _gold_evidence_for_case(corpus, case)
    claims = build_expected_claims(corpus, case)
    return GoldenCase(
        id=case.id,
        question=case.question,
        category=case.category,
        test_case=case,
        golden_context=[e.evidence_text for e in evidence],
        golden_evidence_ids=[e.evidence_id for e in evidence],
        golden_answer=build_golden_answer(corpus, case, evidence, claims),
        expected_claims=claims,
    )


def build_golden_dataset(cases: list[TestCase], corpus: GoldCorpus) -> list[GoldenCase]:
    return [build_golden_case(c, corpus) for c in cases]


def golden_retrieval_documents(golden: GoldenCase, retrieval_documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
    """The REAL RetrievalDocument objects (the same ones production
    uses) whose evidence_id is in this case's golden set -- selected by
    gold membership, not by the retrieval ranking algorithm, so
    generator_eval.py can call the real generation hook on known-good
    context without reimplementing RetrievalDocument construction (same
    technique runner.py's own poisoned-evidence test already uses)."""
    wanted = set(golden.golden_evidence_ids)
    return [d for d in retrieval_documents if d.evidence_id in wanted]
