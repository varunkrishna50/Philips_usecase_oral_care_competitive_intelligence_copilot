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


# ---------------------------------------------------------------------------------------------------
# Negative control: a checker that cannot fail proves nothing, so run it on sentences that are
# deliberately false (it must flag each) and on true ones (it must stay silent).
# ---------------------------------------------------------------------------------------------------

def _unsupported_number(corpus: GoldCorpus, product: GoldProduct, candidates: list[str]) -> str:
    """The first candidate number that appears nowhere in the product's evidence or attribute values."""
    texts = [e.evidence_text for e in corpus.evidence if e.product_id == product.product_id]
    known = _gold_value_strings(product)
    for number in candidates:
        if number not in known and not any(number in t for t in texts):
            return number
    raise ValueError(f"every candidate number is supported for {product.product_name}")


def run_negative_controls(corpus: GoldCorpus) -> dict:
    """Feeds the checker deliberately false and known-true sentences built from the gold corpus.

    False (each must be flagged): a made-up mode count, a made-up price, a made-up percentage,
    a denied feature the product has, and a sentence about another product under a one-product
    scope. True (each must be clean): the product's real price, an affirmed feature it has.
    """
    with_price = [p for p in corpus.products if isinstance(p.get("price"), (int, float)) and not isinstance(p.get("price"), bool)]
    with_pressure = [p for p in corpus.products if p.get("pressure_sensing") is True]
    if len(corpus.products) < 2 or not with_price or not with_pressure:
        raise ValueError("gold corpus too small for the negative control")
    product, other = with_price[0], next(p for p in corpus.products if p.product_id != with_price[0].product_id)
    pressure_product = with_pressure[0]

    modes = _unsupported_number(corpus, product, ["42", "47", "83", "91"])
    price = _unsupported_number(corpus, product, ["7.77", "1234.56", "8.88"])
    pct = _unsupported_number(corpus, product, ["8123", "7311", "6421"])

    items: list[dict] = []

    def record(label: str, text: str, expect_flag: bool, flagged: bool) -> None:
        items.append({"label": label, "text": text, "expect_flag": expect_flag, "flagged": flagged,
                      "ok": flagged == expect_flag})

    for label, text in (
        ("made-up mode count", f"{product.product_name} has {modes} brushing modes."),
        ("made-up price", f"{product.product_name} costs ${price}."),
        ("made-up percentage", f"{product.product_name} removes {pct}% more plaque."),
    ):
        record(label, text, True, bool(check_segment(text, corpus).issues))

    denial = f"{pressure_product.product_name} does not have a pressure sensor."
    record("denies a feature the product has", denial, True,
           check_boolean_claim(denial, pressure_product, "pressure_sensing", True) is not None)

    wrong = f"{other.product_name} has a timer."
    record("names a product outside the question's scope", wrong, True,
           bool(check_segment(wrong, corpus, allowed_product_ids={product.product_id}).issues))

    true_price = f"{product.product_name} costs ${float(product.get('price')):.2f}."
    record("true price", true_price, False, bool(check_segment(true_price, corpus, {product.product_id}).issues))

    affirm = f"{pressure_product.product_name} has a pressure sensor."
    record("true feature", affirm, False,
           check_boolean_claim(affirm, pressure_product, "pressure_sensing", True) is not None)

    false_items = [i for i in items if i["expect_flag"]]
    true_items = [i for i in items if not i["expect_flag"]]
    return {
        "false_sentences": len(false_items),
        "false_flagged": sum(1 for i in false_items if i["flagged"]),
        "true_sentences": len(true_items),
        "true_clean": sum(1 for i in true_items if not i["flagged"]),
        "all_ok": all(i["ok"] for i in items),
        "items": items,
    }
