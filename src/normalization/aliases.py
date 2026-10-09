"""Product aliases and alias groups.

* ``build_aliases`` - names people type ("3000 series", "HX3681", "iO6") ->
  product_id. Written to ``product_aliases.csv``; the query matcher that will
  use it is deferred (no ``src/rag`` edits in this pass).
* ``alias_groups`` - products whose pages are content-identical (iO9 and
  iO9 Genius: same copy, different URL/colour) share an ``alias_group``; the
  canonical member is the one without an edition word.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable
from urllib.parse import urlparse

EDITION_WORDS = {"genius", "pro", "plus", "deluxe", "kids", "limited", "special"}
IDENTICAL_RATIO = 0.95


def _norm(alias: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", alias.lower()).strip()


def _slug_words(url: str) -> list[str]:
    return [w for w in re.split(r"[-_/]+", urlparse(url).path.lower()) if w]


def build_aliases(products: list[dict]) -> list[dict]:
    """products = [{product_id, brand, product_name, product_url, canonical_product_id}]"""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(alias: str, kind: str, p: dict) -> None:
        alias = alias.strip()
        norm = _norm(alias)
        if len(norm) < 3 or (norm, p["product_id"]) in seen:
            return
        seen.add((norm, p["product_id"]))
        rows.append(
            {
                "alias": alias, "normalized_alias": norm, "alias_type": kind,
                "product_id": p["product_id"], "canonical_product_id": p.get("canonical_product_id") or p["product_id"],
            }
        )

    for p in products:
        pid, name, url, brand = p["product_id"], p["product_name"], p["product_url"], p["brand"]
        add(pid, "product_id", p)
        add(name, "name", p)
        stem = pid.split("_", 1)[1]
        add(stem.replace("-", " "), "slug", p)

        if brand == "Philips":
            m = re.match(r"([a-z]{2}\d{4})-(\d{2})", stem)
            if m:
                model, suffix = m.group(1).upper(), m.group(2)
                for a in (model, f"{model}/{suffix}", f"{model}-{suffix}", f"{model}_{suffix}", f"{model} {suffix}"):
                    add(a, "model_number", p)
            # series names from name and URL slug: "3100 series", "sonicare 3000 series",
            # "DiamondClean Smart 9300", "Prestige 9900", "DiamondClean 9000"
            numbers: dict[str, str] = {}  # series number -> "series" (from the title) or "series_from_url_slug"
            for source, kind in ((name, "series"), (" ".join(_slug_words(url)), "series_from_url_slug")):
                for m2 in re.finditer(r"(\d{3,4})[ -]series|series[ -](\d{3,4})|\b([3-9]\d00)\b", source, re.I):
                    numbers.setdefault(m2.group(1) or m2.group(2) or m2.group(3), kind)
            for num, kind in sorted(numbers.items()):
                add(f"{num} series", kind, p)
                add(f"Sonicare {num}", kind, p)
                add(f"Sonicare {num} series", kind, p)
                add(f"Series {num}", kind, p)
            low_name = name.lower()
            for family, aliases in (
                ("diamondclean smart", ("DiamondClean Smart", "Sonicare DiamondClean Smart")),
                ("prestige", ("Prestige", "Sonicare Prestige", "DiamondClean Prestige")),
                ("diamondclean", ("DiamondClean", "Sonicare DiamondClean")),
            ):
                if family in low_name and not (family == "diamondclean" and "diamondclean smart" in low_name):
                    for a in aliases:
                        add(a, "family", p)
            if re.search(r"\bone for kids\b", low_name):
                for a in ("One For Kids", "Sonicare One For Kids", "Philips One For Kids", "kids toothbrush"):
                    add(a, "series", p)
            elif re.search(r"\bphilips one\b|\bone by sonicare\b", low_name):
                for a in ("Philips One", "Sonicare One", "One by Sonicare", "Philips One by Sonicare", "HY1200"):
                    add(a, "series", p)
        else:
            nums = re.findall(r"\bi?o\s*series[ -]?(\d+)|\bio[ -]?(\d+)\b|series[ -](\d+)", name + " " + stem.replace("-", " "), re.I)
            for tup in nums:
                n = next((x for x in tup if x), None)
                if n:
                    for a in (f"iO{n}", f"iO {n}", f"iO Series {n}", f"io series {n}", f"Oral-B iO{n}", f"Series {n}"):
                        add(a, "series", p)
            if "genius" in stem:
                add("Genius", "edition", p)
            # variant words after the series number ("clean gum care", "clean protect"):
            # "iO4 Clean Gum Care" tells the two iO4 pages apart
            m3 = re.search(r"series-(\d+)-(.*)$", stem)
            if m3:
                extra = [w for w in m3.group(2).split("-") if w not in _GENERIC_WORDS]
                if extra:
                    n, tail = m3.group(1), " ".join(extra)
                    for a in (f"iO{n} {tail}", f"iO {n} {tail}", f"iO Series {n} {tail}", f"Oral-B iO{n} {tail}"):
                        add(a, "variant", p)
                    if "gum" in extra and "care" in extra:
                        for a in (f"iO{n} Clean & Gum Care", f"iO Series {n} Clean & Gum Care"):
                            add(a, "variant", p)
    _mark_ambiguous(rows, products)
    return rows


_GENERIC_WORDS = {"rechargeable", "electric", "toothbrush", "starter", "kit", "genius"}


def _mark_ambiguous(rows: list[dict], products: list[dict]) -> None:
    """An alias that resolves to more than one canonical product (the bare
    "iO4" matches two different pages) is flagged and lists the candidates,
    so a matcher asks instead of guessing."""
    names = {p["product_id"]: p["product_name"] for p in products}
    canon_by_alias: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        canon_by_alias[r["normalized_alias"]].add(r["canonical_product_id"])
    slug_only = {(r["normalized_alias"], r["product_id"]) for r in rows if r["alias_type"] == "series_from_url_slug"}
    for r in rows:
        cands = canon_by_alias[r["normalized_alias"]]
        r["ambiguous"] = "true" if len(cands) > 1 else ""
        r["candidates"] = " | ".join(sorted(f"{c} ({names.get(c, '')})" for c in cands)) if len(cands) > 1 else ""
        if len(cands) > 1:
            # a number that only one product's URL slug carries (not its title) explains the clash
            notes = [
                f"{pid.split('_', 1)[1]} is titled '{names.get(pid, '')}' but its URL slug says '{r['normalized_alias']}'"
                for (alias, pid) in sorted(slug_only) if alias == r["normalized_alias"]
            ]
            if notes:
                r["candidates"] += " | NOTE: " + "; ".join(notes)


def _fingerprint_ratio(a: dict, b: dict, keys: Iterable[str]) -> float:
    keys = list(keys)
    same = sum(1 for k in keys if a.get(k) == b.get(k))
    return same / len(keys) if keys else 0.0


def alias_groups(flat_rows: list[dict[str, Any]], attribute_keys: list[str]) -> dict[str, dict]:
    """flat_rows = [{product_id, <attribute values...>}] -> {product_id: {alias_group, canonical_product_id}}.
    Rows whose attribute values are >= 95% identical form a group."""
    parent: dict[str, str] = {r["product_id"]: r["product_id"] for r in flat_rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(flat_rows):
        for b in flat_rows[i + 1:]:
            if a["product_id"].split("_", 1)[0] != b["product_id"].split("_", 1)[0]:
                continue  # never group across brands
            if _fingerprint_ratio(a, b, attribute_keys) >= IDENTICAL_RATIO:
                parent[find(a["product_id"])] = find(b["product_id"])

    groups: dict[str, list[str]] = defaultdict(list)
    for r in flat_rows:
        groups[find(r["product_id"])].append(r["product_id"])

    out: dict[str, dict] = {}
    for members in groups.values():
        if len(members) == 1:
            out[members[0]] = {"alias_group": "", "canonical_product_id": members[0]}
            continue
        base = [m for m in members if not (set(re.split(r"[-_]", m)) & EDITION_WORDS)]
        canonical = min(base or members, key=lambda m: (len(m), m))
        group_id = "grp_" + canonical.split("_", 1)[1]
        for m in members:
            out[m] = {"alias_group": group_id, "canonical_product_id": canonical}
    return out
