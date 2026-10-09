"""Deterministic scope guardrails for 3b generation.

Found via manual testing: queries that name an out-of-scope source (a
third-party retailer, a non-US regional page) or that carry no
recognizable product/attribute at all (open-ended advice questions
included) were falling through to the generic "dump top-k retrieved
evidence" branch of `_answer_normal`, which is only safe when the
retrieval was actually scoped by product or attribute. Unscoped, it
surfaces whatever ranked highest regardless of relevance -- including
raw/truncated structured-data fragments in a couple of observed cases.
This module adds a cheap, deterministic (no LLM, no corpus lookup)
pre-check so those queries get an explicit, honest scope statement
instead of a pseudo-answer built from irrelevant evidence.
"""
from __future__ import annotations

import re
from typing import Optional

_THIRD_PARTY_RETAILERS = (
    "amazon", "ebay", "walmart", "target", "best buy", "bestbuy", "costco", "ulta",
)

_NON_US_MARKERS = (
    "uk page", "uk site", "uk website", " uk ", "united kingdom", "u.k.",
    "europe", "european", " eu ", "canada", "canadian", "australia", "australian",
    "india", "indian", "germany", "german site", "france", "french site",
)


def detect_out_of_scope_source(query: str) -> Optional[str]:
    """Returns a refusal message if the query asks about a source outside
    the approved corpus (3rd-party retailer pricing/listings, or a
    non-US regional page), else None. Checked before retrieval runs."""
    lowered = f" {query.lower()} "

    for marker in _THIRD_PARTY_RETAILERS:
        if marker in lowered:
            return (
                "I can only answer from the official Philips US (www.usa.philips.com) and Oral-B US "
                "(oralb.com) product pages, so I can't help with prices or listings from other retailers."
            )

    for marker in _NON_US_MARKERS:
        if marker in lowered:
            return (
                "I can only answer from the official US product pages (www.usa.philips.com and oralb.com), "
                "so I can't help with pages from other countries."
            )

    return None


# Medical/dental-advice detection (found via 3C.1 live evaluation: a
# question naming a REAL product -- "Is the iO Series 4 Clean & Gum
# Care good for gum disease?" -- has product_match.kind=="matched",
# so detect_out_of_scope_source/has_no_scoping_signal never trigger for
# it; it falls through to the generic evidence branch and reaches the
# LLM, which (live) returned no usable segments for this one. This is
# the SAME deterministic short-circuit mechanism as the rest of this
# module, just a new, narrow trigger.
#
# Conservative by design: triggers only on an explicit advice-seeking
# VERB ("good for", "treat", "cure", "heal", "safe for") co-occurring
# with a named dental/medical CONDITION ("gum disease", "gingivitis",
# ...), or a handful of standalone phrases ("medically recommended")
# that are unambiguous on their own. A pure claims/feature question --
# "What does the manufacturer claim about gum health?", "Does this
# have a gum-care mode?" -- contains neither an advice verb nor (in the
# first case) a named condition, so it is never caught by this check.
_MEDICAL_ADVICE_VERB_RE = re.compile(r"\b(good for|treat|cure|heal|safe for)\b", re.I)
_MEDICAL_CONDITION_RE = re.compile(r"gum disease|gingivitis|periodontal|cavit(?:y|ies)|tooth decay|oral disease", re.I)
_STANDALONE_MEDICAL_ADVICE_RE = re.compile(
    r"medically recommended|medical advice|medical recommendation|dental advice", re.I
)

MEDICAL_ADVICE_MESSAGE = (
    "I can't give medical or dental advice, or say whether a product treats, cures, or is "
    "recommended for a health condition. I can share what the manufacturer's own product page "
    "claims (for example its plaque-removal or gum-health claims). Please ask about a claim "
    "or a specific feature instead."
)


def detect_medical_advice_request(query: str) -> Optional[str]:
    """Returns MEDICAL_ADVICE_MESSAGE if the query is asking for a
    medical/dental recommendation, diagnosis, or treatment judgment
    rather than a product fact or published claim; else None. Checked
    before retrieval runs, same as detect_out_of_scope_source, so a
    named product can never route a medical-advice question to the LLM."""
    if _STANDALONE_MEDICAL_ADVICE_RE.search(query):
        return MEDICAL_ADVICE_MESSAGE
    if _MEDICAL_ADVICE_VERB_RE.search(query) and _MEDICAL_CONDITION_RE.search(query):
        return MEDICAL_ADVICE_MESSAGE
    return None


_WORD_RE = re.compile(r"[a-z0-9]+")


def has_no_scoping_signal(product_match_kind: str, attributes: list[str]) -> bool:
    """True when the query gave neither a specific product nor a
    specific documented attribute to scope retrieval by -- the case
    where falling back to raw top-k hybrid results is unsafe because
    nothing constrained their relevance."""
    return product_match_kind != "matched" and not attributes


NO_SCOPE_MESSAGE = (
    "I couldn't tell which product or which feature you mean (for example price, battery life, "
    "modes or pressure sensing). I can only answer questions about what the official Philips US "
    "and Oral-B US product pages say, not general recommendations or advice. Please ask about a "
    "specific product or feature."
)
