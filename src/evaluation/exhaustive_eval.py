"""3C.4/3C.9: exhaustive-query completeness, measured against an
expected product set filtered directly from comparison_table.parquet
(via gold_labels.py), never through RetrievalEngine/corpus.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gold_labels import GoldCorpus


def gold_exhaustive_expected(corpus: GoldCorpus, attribute: str, condition_value: Any = True) -> set[str]:
    return {p.product_id for p in corpus.products_where(attribute, condition_value)}


@dataclass
class ExhaustiveCheckResult:
    attribute: str
    expected_count: int
    actual_count: int
    recall: float
    missing_product_ids: list[str] = field(default_factory=list)
    extra_product_ids: list[str] = field(default_factory=list)
    every_match_has_evidence: bool = True

    @property
    def complete(self) -> bool:
        return self.recall == 1.0 and not self.extra_product_ids and self.every_match_has_evidence


def check_exhaustive_completeness(expected_product_ids: set[str], actual_matches: list) -> ExhaustiveCheckResult:
    """actual_matches: list of objects with `.product.product_id` and
    `.evidence` (src.rag.exhaustive.ExhaustiveMatch), i.e. the pipeline's
    OWN output -- compared against the independently-derived expected set."""
    actual_ids = {m.product.product_id for m in actual_matches}
    missing = expected_product_ids - actual_ids
    extra = actual_ids - expected_product_ids
    recall = (len(expected_product_ids & actual_ids) / len(expected_product_ids)) if expected_product_ids else 1.0
    every_has_evidence = all(len(m.evidence) >= 1 for m in actual_matches)
    return ExhaustiveCheckResult(
        attribute="", expected_count=len(expected_product_ids), actual_count=len(actual_ids),
        recall=recall, missing_product_ids=sorted(missing), extra_product_ids=sorted(extra),
        every_match_has_evidence=every_has_evidence,
    )
