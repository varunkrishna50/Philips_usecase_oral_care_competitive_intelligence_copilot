"""Content checks for test cases: phrases an answer must (``must_include``) or must not
(``must_not_include``) contain.

Matching is on the answer text after a comparison-only normalisation (case, curly quotes and
dashes, whitespace, a ``$`` before a figure, thousands commas). A phrase that is a figure
("5x", "700%", "69.99") must match as a whole token: it may not start or end inside a longer
number or word, so "5x" is not found in "15x" and "69.99" is not found in "$169.99".
Nothing here changes the answer; stored text keeps its original characters.
"""
from __future__ import annotations

import re
from typing import Any

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x00A0: " "}


def normalize(text: str) -> str:
    text = text.translate({**_DASHES, **_QUOTES}).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 62,000 -> 62000
    text = re.sub(r"\$\s*(?=\d)", "", text)  # $69.99 -> 69.99
    return re.sub(r"\s+", " ", text).strip()


def _is_figure(phrase: str) -> bool:
    return bool(re.search(r"\d", phrase))


def phrase_found(answer: str, phrase: str) -> bool:
    haystack, needle = normalize(answer), normalize(phrase)
    if not needle:
        return False
    if _is_figure(needle):
        # not preceded or followed by a word character or by a digit-joining "."/"," ("169.99", "15x")
        pattern = r"(?<![\w.,])" + re.escape(needle) + r"(?![\w])(?!\.\d)(?!,\d)"
        return re.search(pattern, haystack) is not None
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


def check_content(answer: str, must_include: list[str], must_not_include: list[str]) -> list[dict[str, Any]]:
    """[{phrase, kind, found, ok}] in order: every must_include, then every must_not_include."""
    checks = []
    for phrase in must_include:
        found = phrase_found(answer, phrase)
        checks.append({"phrase": phrase, "kind": "must_include", "found": found, "ok": found})
    for phrase in must_not_include:
        found = phrase_found(answer, phrase)
        checks.append({"phrase": phrase, "kind": "must_not_include", "found": found, "ok": not found})
    return checks


def summarize(checks: list[dict[str, Any]]) -> str:
    """'haptic found, gentle vibrations found, 399.99 absent' / '... 5x MISSING' / '... 9300 PRESENT (forbidden)'"""
    parts = []
    for c in checks:
        if c["kind"] == "must_include":
            parts.append(f"{c['phrase']} found" if c["found"] else f"{c['phrase']} MISSING")
        else:
            parts.append(f"{c['phrase']} PRESENT (forbidden)" if c["found"] else f"{c['phrase']} absent")
    return ", ".join(parts)
