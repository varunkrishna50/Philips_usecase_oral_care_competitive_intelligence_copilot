"""Mode-name vocabulary: map each brand's wording to a canonical key with an
*equivalence* flag, so brands can be compared without pretending different
modes are the same.

* ``exact``       - the source name IS the canonical mode (or only differs by
                    spelling/hyphenation, e.g. Super-Sensitive = Super Sensitive).
* ``approximate`` - a related but not identical mode (Whitening / White / White+
                    vs Whiten; Gum Health vs Gum Care; Daily Clean vs Clean;
                    Intense Clean vs Intense). A comparison should say "similar",
                    never "same".

``Sensitive`` and ``Super Sensitive`` are different keys. Modes with no
mapping keep their own slug as canonical key with ``exact``.
"""
from __future__ import annotations

import re
from typing import Optional

# (pattern on normalized name, canonical key, equivalence)
_RULES: list[tuple[str, str, str]] = [
    (r"^super[ -]?sensitive$", "super_sensitive", "exact"),
    (r"^sensitive$", "sensitive", "exact"),
    (r"^whiten$", "whiten", "exact"),
    (r"^(?:whitening|white\+?|ultimate white)$", "whiten", "approximate"),
    (r"^gum care$", "gum_care", "exact"),
    (r"^gum health$", "gum_care", "approximate"),
    (r"^clean$", "clean", "exact"),
    (r"^daily clean$", "clean", "approximate"),
    (r"^intense$", "intense", "exact"),
    (r"^intense clean$", "intense", "approximate"),
    (r"^deep clean\+?$", "deep_clean", "exact"),
    (r"^tongue clean$", "tongue_clean", "exact"),
    (r"^gentle$", "gentle", "exact"),
]
_COMPILED = [(re.compile(p), key, eq) for p, key, eq in _RULES]


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("‑", "-").replace("‐", "-").strip().lower())


def canonical_mode(name: str) -> dict:
    n = _norm(name)
    for rx, key, eq in _COMPILED:
        if rx.match(n):
            return {"name": name, "canonical": key, "equivalence": eq}
    slug = re.sub(r"[^a-z0-9]+", "_", n).strip("_") or "unknown"
    return {"name": name, "canonical": slug, "equivalence": "exact"}


def canonicalize_modes(modes: list[dict]) -> list[dict]:
    """modes = [{name, description}] -> [{name, canonical, equivalence, description}]."""
    out = []
    for m in modes:
        entry = canonical_mode(m["name"])
        entry["description"] = m.get("description")
        out.append(entry)
    return out


def compare_mode_names(a: str, b: str) -> str:
    """'same' | 'similar' | 'different' - for the (deferred) comparison engine."""
    ca, cb = canonical_mode(a), canonical_mode(b)
    if ca["canonical"] != cb["canonical"]:
        return "different"
    if ca["equivalence"] == "exact" and cb["equivalence"] == "exact":
        return "same"
    return "similar"
