"""3C.7: independent, deterministic groundedness checker.

Deliberately does NOT import src.rag.citations -- it must be able to
catch a bug citations.py itself would miss, and it must check EVERY
segment of every answer, including ones the generation pipeline marked
`trusted` (which citations.py always accepts without inspection). Takes
only plain segment text plus the independently-loaded gold corpus
(gold_labels.py); never touches RetrievalEngine or the RAG corpus
objects.

Limitations, stated plainly rather than overclaimed: numeric and
product-name checks are fully general (regex + substring matching
against gold values/evidence text). Boolean-attribute claims (does the
text correctly say a product has/lacks pressure sensing?) use a
keyword + negation heuristic, not a parser -- good enough to catch an
obviously wrong boolean claim in a short generated sentence, not a
substitute for real NLP. This is called out in the evaluation report,
not hidden.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .comparison_eval import BOOLEAN_ATTRIBUTES, GoldComparison
from .gold_labels import GoldCorpus, GoldProduct

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_NEGATION_RE = re.compile(
    r"\b(not|no|n't|lacks?|without|never|doesn't|does not|isn't|is not)\b", re.I
)
# A number immediately following "a difference of" is a COMPUTED value
# (e.g. a comparison's price/mode-count gap), not a raw fact that can
# appear verbatim in any single evidence_text or attribute value --
# check_comparison_claims() validates it against an independently
# computed gold diff instead; the generic numeric check below must not
# double-flag it as "unsupported".
_DIFFERENCE_NUMBER_RE = re.compile(r"difference of \$?(\d+(?:\.\d+)?)", re.I)

ATTRIBUTE_KEYWORDS = {
    "pressure_sensing": ("pressure",),
    "app_connectivity": ("app", "mobile app"),
    "bluetooth_connectivity": ("bluetooth",),
    "timer": ("timer",),
}


@dataclass
class GroundednessIssue:
    kind: str
    detail: str


@dataclass
class SegmentGroundedness:
    text: str
    issues: list[GroundednessIssue] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.issues


def mentioned_products(text: str, corpus: GoldCorpus) -> list[GoldProduct]:
    lowered = text.lower()
    return [p for p in corpus.products if p.product_name.lower() in lowered]


def _gold_value_strings(product: GoldProduct) -> set[str]:
    out: set[str] = set()
    for attr, value in product.attributes.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out.add(str(value))
            if attr == "price":
                out.add(f"{value:.2f}")
    return out


def check_segment(
    text: str,
    corpus: GoldCorpus,
    allowed_product_ids: Optional[set[str]] = None,
) -> SegmentGroundedness:
    """Checks ONE segment's text in isolation. Caller passes every
    segment of an answer (trusted or not) -- this function has no
    concept of "trusted" at all, by design."""
    issues: list[GroundednessIssue] = []
    mentioned = mentioned_products(text, corpus)

    if allowed_product_ids is not None:
        for p in mentioned:
            if p.product_id not in allowed_product_ids:
                issues.append(GroundednessIssue(
                    "unsupported_product_name",
                    f"text mentions {p.product_name!r}, not among the expected products for this question",
                ))

    mentioned_name_numbers = {n for p in mentioned for n in _NUMBER_RE.findall(p.product_name)}
    difference_numbers = set(_DIFFERENCE_NUMBER_RE.findall(text))
    numbers = [
        n for n in _NUMBER_RE.findall(text)
        if n not in mentioned_name_numbers and n not in difference_numbers
    ]

    if mentioned and numbers:
        # A number is supported if it's backed by ANY ONE of the
        # mentioned products -- a multi-product segment (comparisons,
        # exhaustive lines naming more than one product) legitimately
        # states a fact about product B that won't appear in product
        # A's own evidence/attributes; checking each number against
        # every mentioned product individually would false-flag that.
        all_evidence_texts = [e.evidence_text for e in corpus.evidence if e.product_id in {p.product_id for p in mentioned}]
        all_gold_values: set[str] = set()
        for p in mentioned:
            all_gold_values |= _gold_value_strings(p)

        for number in numbers:
            supported = number in all_gold_values or any(number in t for t in all_evidence_texts)
            if not supported:
                kind = "unsupported_price" if "$" in text else "unsupported_numeric_value"
                names = ", ".join(p.product_name for p in mentioned)
                issues.append(GroundednessIssue(
                    kind,
                    f"number {number!r} is not in any evidence_text or normalized attribute value for {names}",
                ))

    return SegmentGroundedness(text=text, issues=issues)


def check_boolean_claim(text: str, product: GoldProduct, attribute: str, gold_value: Optional[bool]) -> Optional[GroundednessIssue]:
    """Heuristic: does the claimed polarity (has/lacks <attribute>) in
    `text` match the gold boolean value for `product`? Returns None if
    the attribute keyword isn't mentioned at all (nothing to check) or
    gold_value is None (not_mentioned/ambiguous -- no ground truth to
    check against)."""
    if gold_value is None:
        return None
    keywords = ATTRIBUTE_KEYWORDS.get(attribute, (attribute.replace("_", " "),))
    lowered = text.lower()
    if not any(kw in lowered for kw in keywords):
        return None
    # Look for a negation within a short window before the first keyword hit.
    first_kw_pos = min(lowered.find(kw) for kw in keywords if kw in lowered)
    window = lowered[max(0, first_kw_pos - 30):first_kw_pos]
    negated = bool(_NEGATION_RE.search(window))
    claimed_true = not negated
    if claimed_true != gold_value:
        return GroundednessIssue(
            "unsupported_boolean_attribute",
            f"text claims {attribute}={claimed_true} for {product.product_name}, gold value is {gold_value}",
        )
    return None


def check_comparison_claims(text: str, gold: GoldComparison, product_a: GoldProduct, product_b: GoldProduct) -> list[GroundednessIssue]:
    """Flags a stated numeric difference that doesn't match the
    independently-computed gold difference (3C.7's "unsupported
    comparisons")."""
    issues: list[GroundednessIssue] = []
    if gold.attribute not in BOOLEAN_ATTRIBUTES and gold.comparable and gold.diff is not None:
        match = re.search(r"difference of \$?(\d+(?:\.\d+)?)", text)
        if match:
            claimed_diff = float(match.group(1))
            if abs(claimed_diff - gold.diff) > 1e-6:
                issues.append(GroundednessIssue(
                    "unsupported_comparison",
                    f"text claims a difference of {claimed_diff} for {gold.attribute}, gold diff is {gold.diff}",
                ))
    return issues
