"""Section-exclusion rules shared by both brands.

Attribute extraction (utils.py, philips.py, oralb.py) MUST only ever scan
the output of usable_sections() -- never record["raw_text"] directly --
so that reviews, cross-sell, newsletter, and cross-product comparison
content can never silently become evidence for the current product. This
single chokepoint is what makes the "no cross-product contamination" and
"no UGC-as-manufacturer-claim" rules structural rather than something
every extractor has to remember to re-implement.
"""
from __future__ import annotations

import re

from .models import NormalizationWarning

# Exact (case-insensitive) heading matches that are always page chrome /
# non-manufacturer content, on either brand's pages. No warning needed --
# excluding these is routine, expected, and not evidence of a data problem.
EXCLUDED_HEADINGS_EXACT = {
    "sort by",
    "page content",
    "get support for this product",
    "reviews",
    # Verified byte-for-byte identical across all 13 Oral-B products during
    # Phase 1 inspection -- a fixed testimonial-widget placeholder, not
    # product-specific review content.
    "the perfect brush for my teeth",
}

# Heading substrings/patterns identifying cross-sell, accessory-upsell, or
# generic marketing sections that are not about the current product's own
# included contents or specs.
EXCLUDED_HEADING_PATTERNS = [
    re.compile(r"you may also like", re.I),
    re.compile(r"add accessories", re.I),
    re.compile(r"parts\s*&\s*accessories", re.I),
    re.compile(r"find a spare part or an accessory", re.I),
    re.compile(r"sign-?up.*save", re.I),
]

# Headings that compare the current product against a *different* named
# model/series are dropped entirely and loudly (via a warning), even when
# some of their answer text might be safe in isolation -- see the Step 10
# Phase 1 finding: "What's the difference between Series 9 and 10?" mixes
# facts about both models in one answer, and "Does Series 10 have the same
# 7 modes as Series 9?" cannot be reliably told apart from that by regex
# alone. Conservative: exclude both.
COMPARISON_HEADING_PATTERNS = [
    re.compile(r"difference between", re.I),
    re.compile(r"\bcompared to\b", re.I),
    re.compile(r"\bvs\.?\b", re.I),
    re.compile(r"\bversus\b", re.I),
    re.compile(r"\bsame\b.{0,40}\bas\b", re.I),
]


def classify_heading(heading: str) -> tuple[bool, str | None]:
    """Returns (excluded, warning_code). warning_code is None when the
    exclusion is routine chrome-filtering that doesn't need flagging.
    """
    h = heading.strip().lower()
    if h in EXCLUDED_HEADINGS_EXACT:
        return True, None
    for pat in EXCLUDED_HEADING_PATTERNS:
        if pat.search(heading):
            return True, None
    for pat in COMPARISON_HEADING_PATTERNS:
        if pat.search(heading):
            return True, "cross_product_comparison_section_excluded"
    return False, None


def usable_sections(
    sections: list[dict],
    product_id: str,
    warnings: list[NormalizationWarning],
) -> list[tuple[int, dict]]:
    """Returns [(original_index, section_dict), ...] for sections that are
    safe to use as evidence for THIS product, in original page order.
    """
    out: list[tuple[int, dict]] = []
    for i, section in enumerate(sections):
        excluded, warning_code = classify_heading(section["heading"])
        if excluded:
            if warning_code:
                warnings.append(
                    NormalizationWarning(
                        product_id=product_id,
                        code=warning_code,
                        message=f"Excluded comparison-framed section {section['heading']!r} from evidence consideration.",
                        section_heading=section["heading"],
                    )
                )
            continue
        out.append((i, section))
    return out
