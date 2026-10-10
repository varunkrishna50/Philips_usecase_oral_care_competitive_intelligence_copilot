"""Query understanding: product recognition (3A.8/14/15), attribute
mapping (3A.9), and intent classification (3A.13).

Product recognition stays plain tokenization/regex over the real
catalog (a closed, enumerable vocabulary -- no benefit to an LLM call
for exact model/alias matching). Attribute mapping and intent
classification are LLM-first (classify_query) -- open-vocabulary
paraphrase (e.g. "how long do the heads last" for the same thing as
"replacement interval") is exactly what regex enumeration can't keep
up with; see the regex functions kept below (now internal) for the
deterministic fallback used only when the LLM is unavailable/fails.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from .config import PRODUCT_ALIASES_CSV
from .llm_provider import LLMUnavailable, OpenAIStructuredClient
from .models import ProductRecord

ProductMatchKind = Literal["matched", "ambiguous", "not_in_corpus", "none"]
Intent = Literal["exhaustive", "comparison", "normal"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Brand/category words so common across the catalog that matching them
# alone says nothing about which specific product is meant -- e.g. some
# Oral-B product names include an "Oral-B" prefix and some don't, so a
# raw "longest contiguous match" would let a generic "Oral-B iO Series"
# prefix (shared by several products) outscore a short but genuinely
# unique "iO Series 10" match. Scoring discriminative-token count FIRST,
# raw length only as a tiebreaker, fixes that.
_GENERIC_MATCH_TOKENS = {
    "oral", "b", "oralb", "io", "series", "electric", "toothbrush",
    "toothbrushes", "rechargeable", "philips", "sonicare", "sonic",
}


def _discriminative_count(tokens: list[str]) -> int:
    return sum(1 for t in tokens if t not in _GENERIC_MATCH_TOKENS)


def _best_match_score(query_tokens: list[str], target_tokens: list[str]) -> tuple[int, int]:
    """(discriminative_token_count, raw_length) of the best contiguous
    span of query_tokens that also appears, in order, in target_tokens.
    Compared as a tuple, so a match carrying a real distinguishing token
    (a model number, "genius", "prestige", ...) always outranks a longer
    but purely generic brand/category-word match."""
    if not query_tokens or not target_tokens:
        return (0, 0)
    target_str = " " + " ".join(target_tokens) + " "
    best = (0, 0)
    n = len(query_tokens)
    for i in range(n):
        for j in range(i + 1, n + 1):
            span = query_tokens[i:j]
            phrase = " " + " ".join(span) + " "
            if phrase in target_str:
                score = (_discriminative_count(span), j - i)
                if score > best:
                    best = score
    return best


@dataclass
class ProductMatch:
    kind: ProductMatchKind
    candidates: list[ProductRecord] = field(default_factory=list)
    match_len: int = 0


# Brand/family words common enough that a query built ONLY from these
# (plus an unrecognized number) still plausibly references a product --
# used only by the not-in-corpus heuristic below, never to narrow matches.
_BRAND_WORDS = {"philips", "sonicare", "oral", "oralb", "io"}
_COMPETITOR_BRANDS = {
    "colgate", "quip", "waterpik", "burst", "goby", "oclean", "fairywill", "suri",
}

MIN_PRODUCT_MATCH_LEN = 2
MAX_GENERIC_TIE = 20  # a tie this wide is a category phrase, not a product family


def _pre_color_discriminative_tokens(product_name: str) -> set[str]:
    """Discriminative tokens from the model-identifying part of a
    product name only -- the part BEFORE a trailing ", Color Name"
    (every product in this catalog that has a color variant follows
    that convention, confirmed across the real corpus: "...Rechargeable
    Electric Toothbrush, Aqua Marine", "...Genius Rechargeable Electric
    Toothbrush, Sapphire Blue", etc.). A color is almost never mentioned
    by a user and must never count as an "extra" qualifier that makes
    an otherwise-exact model match look ambiguous against a same-family
    variant that has a real, functionally-different qualifier word
    (e.g. "iO Series 9" vs "iO Series 9 Genius" -- the distinguishing
    word there is "Genius", not the color, and that's what should drive
    disambiguation when the query only ever says "iO Series 9")."""
    pre_comma = product_name.split(",", 1)[0]
    return {t for t in normalize_tokens(pre_comma) if t not in _GENERIC_MATCH_TOKENS}


def _known_product_numbers(products: list[ProductRecord]) -> set[str]:
    known: set[str] = set()
    for p in products:
        known.update(t for t in normalize_tokens(p.product_id) + normalize_tokens(p.product_name) if t.isdigit())
    return known


def _detect_products_by_name(query: str, products: list[ProductRecord]) -> ProductMatch:
    """Deterministic product recognition over the actual catalog (3A.8).

    - A unique best-scoring match -> "matched" (exact product/model
      identifiers always win over a semantically similar but different
      model, since this never consults embeddings; see
      _best_match_score for why raw phrase length alone isn't enough).
    - A tie confined to ONE brand -> "ambiguous" (e.g. "iO Series" with
      no number matches every Oral-B iO product; "electric toothbrush"
      ties across BOTH brands and is rejected as a generic category
      phrase instead, see MAX_GENERIC_TIE / brand-homogeneity check).
    - The query names a specific model number that doesn't belong to
      ANY real product (so the best match is purely generic brand/family
      words), or names a known competitor brand -> "not_in_corpus" (3A.15).
    - Otherwise -> "none" (no product referenced at all).
    """
    query_tokens = normalize_tokens(query)
    query_numbers = {t for t in query_tokens if t.isdigit()}
    has_unknown_number = bool(query_numbers - _known_product_numbers(products))
    has_brand_word = any(t in _BRAND_WORDS for t in query_tokens)
    has_competitor = any(t in _COMPETITOR_BRANDS for t in query_tokens)

    scored: list[tuple[ProductRecord, tuple[int, int]]] = []
    for p in products:
        id_tokens = normalize_tokens(p.product_id)
        name_tokens = normalize_tokens(p.product_name)
        score = max(
            _best_match_score(query_tokens, id_tokens),
            _best_match_score(query_tokens, name_tokens),
        )
        scored.append((p, score))

    best_score = max((s for _, s in scored), default=(0, 0))
    disc_count, raw_len = best_score

    if has_competitor:
        return ProductMatch(kind="not_in_corpus", candidates=[], match_len=raw_len)

    if raw_len < MIN_PRODUCT_MATCH_LEN:
        if has_brand_word and has_unknown_number:
            return ProductMatch(kind="not_in_corpus", candidates=[], match_len=raw_len)
        return ProductMatch(kind="none", candidates=[], match_len=raw_len)

    # The only thing that matched is generic brand/family wording, and
    # the query named a specific number that belongs to no real product
    # -- the user tried to be specific and named something that doesn't
    # exist, not an ambiguous family reference.
    if disc_count == 0 and has_unknown_number:
        return ProductMatch(kind="not_in_corpus", candidates=[], match_len=raw_len)

    tied = [p for p, s in scored if s == best_score]
    if len(tied) == 1:
        return ProductMatch(kind="matched", candidates=tied, match_len=raw_len)

    # A genuine tie can still be resolved: prefer whichever tied
    # candidate(s) require NO qualifier word the query didn't already say
    # (ignoring color, which a user is never expected to mention). This
    # is what lets a bare "Oral-B iO Series 9" resolve to the plain
    # iO Series 9 instead of staying ambiguous against "iO Series 9
    # Genius" -- the query never said "genius", so that variant needs
    # something extra the user didn't ask for. If the query DOES name
    # the extra qualifier (e.g. "iO Series 9 Genius"), _best_match_score
    # already gives that candidate a strictly higher score earlier, so
    # this narrowing only ever runs on real ties.
    matched_discriminative = {t for t in query_tokens if t not in _GENERIC_MATCH_TOKENS}
    exact_tied = [
        p for p in tied
        if _pre_color_discriminative_tokens(p.product_name) <= matched_discriminative
    ]
    if len(exact_tied) == 1:
        return ProductMatch(kind="matched", candidates=exact_tied, match_len=raw_len)

    brands = {p.brand for p in tied}
    if len(brands) == 1 and len(tied) <= MAX_GENERIC_TIE:
        return ProductMatch(kind="ambiguous", candidates=tied, match_len=raw_len)
    # Spans multiple brands (or is nearly the whole catalog): a generic
    # category phrase like "electric toothbrush", not a specific family.
    return ProductMatch(kind="none", candidates=[], match_len=raw_len)


# ---------------------------------------------------------------------------
# Alias fallback: names people actually type ("Sonicare 9900", "HX9990", "3100",
# "iO6", "One for Kids") are not always a contiguous run of the catalog name
# ("Sonicare Prestige 9900 Rechargeable Toothbrush"). data/processed/product_aliases.csv
# (written by Step 2) lists them; it is consulted ONLY when the name match found
# no product, so exact name matches and the not-in-corpus rule are unchanged.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _alias_table(path: str = str(PRODUCT_ALIASES_CSV)) -> tuple[dict, ...]:
    """Every alias row of product_aliases.csv as {tokens, product_id, ambiguous, alias_type} (() if there is no file)."""
    file = Path(path)
    if not file.exists():
        return ()
    rows = []
    with file.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            tokens = tuple(normalize_tokens(row.get("normalized_alias", "")))
            if tokens and any(t not in _GENERIC_MATCH_TOKENS for t in tokens):
                rows.append({
                    "tokens": tokens, "product_id": row["product_id"],
                    "ambiguous": row.get("ambiguous") == "true", "alias_type": row.get("alias_type", ""),
                })
    return tuple(rows)


def _load_alias_rows() -> tuple[tuple[tuple[str, ...], str], ...]:
    """((alias tokens), product_id) for every alias in the file ((), if there is none)."""
    return tuple((r["tokens"], r["product_id"]) for r in _alias_table())


def _catalog_names() -> dict[str, tuple[str, ...]]:
    return {r["product_id"]: r["tokens"] for r in _alias_table() if r["alias_type"] == "name"}


def _names_product(tokens: list[str], other_id: str, this_id: str) -> bool:
    """Does the text name ``other_id`` specifically (full name, an unambiguous alias, or a word that
    only that product's name has, such as a colour)?"""
    table = _alias_table()
    names = _catalog_names()
    other_name = names.get(other_id, ())
    if other_name and _contains_phrase(tokens, other_name):
        return True
    if any(r["product_id"] == other_id and not r["ambiguous"] and _contains_phrase(tokens, r["tokens"]) for r in table):
        return True
    distinguishing = {t for t in other_name if t not in set(names.get(this_id, ())) and t not in _GENERIC_MATCH_TOKENS
                      and not t.isdigit()}
    return any(t in tokens for t in distinguishing)


def product_mentioned(text: str, product_id: str, product_name: str) -> bool:
    """True if ``text`` names the product: its full catalog name or a model/series alias.

    A shared alias ("iO4" belongs to two pages) is accepted only when the text does not also name
    another page that shares it, so "the iO4 has an app" counts for the Clean & Gum Care page but an
    answer that also says "Matte White" (the other iO4's colour) does not."""
    tokens = normalize_tokens(text)
    if _contains_phrase(tokens, tuple(normalize_tokens(product_name))):
        return True
    table = _alias_table()
    hit_ambiguous = []
    for row in table:
        if row["product_id"] == product_id and _contains_phrase(tokens, row["tokens"]):
            if not row["ambiguous"]:
                return True
            hit_ambiguous.append(row["tokens"])
    if not hit_ambiguous:
        return False
    others = {r["product_id"] for r in table
              if r["ambiguous"] and r["tokens"] in hit_ambiguous and r["product_id"] != product_id}
    return not any(_names_product(tokens, other, product_id) for other in others)


def _contains_phrase(query_tokens: list[str], phrase: tuple[str, ...]) -> bool:
    n = len(phrase)
    return any(tuple(query_tokens[i:i + n]) == phrase for i in range(len(query_tokens) - n + 1))


def _match_by_alias(query: str, products: list[ProductRecord]) -> ProductMatch | None:
    by_id = {p.product_id: p for p in products}
    query_tokens = normalize_tokens(query)
    best_len, hits = 0, {}
    for tokens, product_id in _load_alias_rows():
        if product_id not in by_id or not _contains_phrase(query_tokens, tokens):
            continue
        if len(tokens) > best_len:
            best_len, hits = len(tokens), {}
        if len(tokens) == best_len:
            hits[product_id] = by_id[product_id]
    if not hits:
        return None
    candidates = sorted(hits.values(), key=lambda p: p.product_id)
    if len(candidates) == 1:
        return ProductMatch(kind="matched", candidates=candidates, match_len=best_len)
    if len({p.brand for p in candidates}) == 1 and len(candidates) <= MAX_GENERIC_TIE:
        return ProductMatch(kind="ambiguous", candidates=candidates, match_len=best_len)
    return None


def detect_products(query: str, products: list[ProductRecord]) -> ProductMatch:
    """Name match first (3A.8); the alias list only when that found no product."""
    match = _detect_products_by_name(query, products)
    if match.kind in ("none", "ambiguous"):
        by_alias = _match_by_alias(query, products)
        if by_alias is not None:
            if match.kind == "none":
                return by_alias
            # "Oral-B iO10" ties every Oral-B page on the brand words alone; the alias narrows it
            # (the tie came from brand/category words only; the alias names a model)
            same_brand = {p.brand for p in by_alias.candidates} == {p.brand for p in match.candidates}
            if same_brand and len(by_alias.candidates) < len(match.candidates):
                return by_alias
    return match


# ---------------------------------------------------------------------------
# Attribute mapping (3A.9) -- natural language -> Step 2 attribute names.
# Every key here must be a real column in comparison_table.csv (see
# src/normalization/models.py:ATTRIBUTE_FIELDS); nothing invented.
# ---------------------------------------------------------------------------

ATTRIBUTE_PATTERNS: dict[str, list[str]] = {
    "price": [r"\bprice\b", r"\bcost\b", r"how much", r"\bexpensive\b", r"\bcheap(?:er)?\b"],
    "mode_count": [r"how many (?:brushing )?modes", r"\bmodes?\b", r"how many settings", r"brushing modes", r"number of modes"],
    "modes": [r"what modes", r"which modes", r"list.*modes", r"names? of.*modes"],
    "pressure_sensing": [
        r"pressure sensor", r"pressure sensing", r"press(?:ing)? too hard",
        r"brush(?:ing)? too hard", r"too much pressure", r"too hard",
        r"excessive pressure", r"detect.*pressure", r"\bpressure\b",
    ],
    "app_connectivity": [r"\bapp\b", r"connect.*(?:phone|app)", r"mobile app", r"smartphone"],
    "bluetooth_connectivity": [r"bluetooth", r"connect.*phone"],
    "timer": [r"\btimer\b", r"2.minute", r"quadpacer", r"smartimer", r"brushing time"],
    "timer_description": [r"\btimer\b", r"how does.*timer"],
    "battery_life_days": [r"battery life", r"how long.*(?:charge|battery)", r"recharge", r"battery last", r"charge last"],
    "battery_life_raw": [r"battery life", r"how long.*(?:charge|battery)", r"battery last"],
    "charging_method": [r"charg(?:e|ing|er)", r"how (?:is|does).*charg"],
    "charging_method_raw": [r"charg(?:e|ing|er)"],
    "included_items": [r"what comes in the box", r"what.?s in the box", r"\bincluded\b", r"come[s]? with", r"package contents", r"what do (?:i|you) get"],
    "description": [r"describe", r"what is (?:the|this)", r"tell me about"],
    "claims": [r"\bclaim", r"marketing claim"],
    "specifications": [r"spec(?:ification)?s?\b"],
    "original_price": [r"original price", r"list price", r"before (?:the )?discount", r"\bmsrp\b", r"regular price"],
    "discount_amount": [r"discount", r"\bon sale\b", r"\bsale\b", r"how much.*(?:off|save)"],
    "discount_pct": [r"discount", r"% off", r"percent(?:age)? off", r"\bon sale\b"],
    "price_after_discount": [r"after (?:the )?discount", r"sale price", r"discounted price"],
    "promotion_text": [r"promo(?:tion)?", r"\boffers?\b", r"\bdeals?\b"],
    "color_count": [r"how many colou?rs", r"number of colou?rs"],
    "colors": [r"\bcolou?rs?\b"],
    "rating_avg": [r"\brating", r"\bstars?\b", r"customer review", r"\breviews?\b"],
    "rating_count": [r"how many (?:ratings|reviews|customers|people)", r"number of (?:ratings|reviews)"],
    "mode_details": [r"describe.*modes?", r"what does.*mode", r"mode descriptions?"],
    "intensity_levels": [r"intensit"],
    "warranty_years": [r"warrant"],
    "battery_type": [r"battery type", r"lithium"],
    "charge_time_hours": [r"charge time", r"how long.*to (?:fully )?charge", r"time to charge", r"fully charge"],
    "brush_movements_per_min": [r"brush(?:ing)? speed", r"movements per minute", r"oscillation", r"vibrations per minute", r"\bspeed\b"],
    "display_type": [r"\bdisplay\b", r"\bscreen\b"],
    "replacement_reminder": [r"replacement reminder", r"brush head.*(?:remind|tracker)", r"when to replace"],
    "coverage_tracking": [r"coverage", r"teeth tracking", r"track.*brushing"],
    "travel_case_included": [r"travel case", r"travel pouch"],
    "brush_heads_included": [r"how many brush heads", r"brush heads? (?:included|come|in the box)"],
    "money_back_guarantee_days": [r"money.back", r"guarantee", r"return policy"],
    "plaque_removal_claim": [r"plaque"],
    "gum_health_claim": [r"gum health", r"healthier gums"],
    "dimensions": [r"\bdimensions?\b", r"\bheight\b", r"how (?:tall|big|large)"],
    "weight": [r"\bweigh", r"\bweight\b", r"how heavy"],
    "disclaimers": [r"disclaimer", r"footnote", r"fine print", r"asterisk", r"compared (?:to|against) what"],
    "availability": [r"in stock", r"out of stock", r"availab", r"can i buy", r"sold out"],
    "listed_price_hidden": [r"hidden price", r"listed price"],
    "pressure_sensing_details": [r"how does the pressure", r"pressure (?:sensor|feedback) (?:work|signal)", r"what happens.*too hard"],
    "app_connectivity_details": [r"how does the app", r"what does the app", r"coaching"],
    "bluetooth_details": [r"how.*bluetooth", r"bluetooth.*(?:work|connect)"],
    "battery_details": [r"battery (?:details|info)", r"low.charge", r"how long.*battery"],
    "charging_details": [r"how (?:do i|to) charge", r"charging (?:details|steps)"],
    "coverage_tracking_details": [r"coverage tracking", r"position detection", r"3d teeth"],
    "brushing_zones": [r"how many (?:brushing )?zones", r"quadrant", r"sextant"],
    "zone_interval_seconds": [r"how often.*(?:alert|vibrat)", r"every \d+ seconds"],
    "brush_head_replacement_interval": [
        r"\b(?:how often|how frequently|when).{0,50}\b(?:replace|change)\b.{0,30}\bbrush heads?\b",
        r"\bbrush heads? replacement interval\b",
        r"\binterval for replacing brush heads?\b",
    ],
    "cleaning_technology": [r"motor", r"cleaning technology", r"micro.vibration", r"oscillat"],
    "brush_head_compatibility": [r"brush head.*compatib", r"compatible.*brush head", r"which brush heads"],
    "feel_and_noise": [r"\bnoise\b", r"\bquiet", r"how does it feel"],
    "price_positioning": [r"price (?:tier|positioning|range)", r"entry.level", r"premium"],
    "noise_db": [r"\bdb\b", r"decibel"],
    "intensity_details": [r"what do.*intensit", r"intensity (?:levels|settings)"],
    "intensity_count": [r"how many intensit"],
    "battery_indicator": [r"battery indicator"],
    "claims_with_disclaimers": [r"what does the .{0,30}claim (?:compare|mean)", r"claim.{0,40}(?:disclaimer|footnote|asterisk)"],
}

_COMPILED_ATTRIBUTE_PATTERNS = {
    attr: [re.compile(p, re.I) for p in patterns] for attr, patterns in ATTRIBUTE_PATTERNS.items()
}


_PRODUCT_NAME_RECHARGEABLE_RE = re.compile(r"\brechargeable\b(?=\s+(?:electric\s+|power\s+|sonic\s+)?(?:tooth)?brush)", re.I)
_DESCRIBE_RE = re.compile(r"\bdescribe\b|tell me about", re.I)


def _detect_attributes_deterministic(query: str) -> list[str]:
    """Regex fallback for detect_attributes(): every Step 2 attribute
    whose natural-language synonyms appear in the query, in
    ATTRIBUTE_PATTERNS's declaration order (stable). Only reached when
    the LLM classifier is unavailable or fails -- see classify_query()."""
    # "Rechargeable" in a product's own name ("...Rechargeable Electric Toothbrush, Cosmic Black")
    # is not a question about battery life or charging.
    query = _PRODUCT_NAME_RECHARGEABLE_RE.sub(" ", query)
    hits = []
    for attr, patterns in _COMPILED_ATTRIBUTE_PATTERNS.items():
        if any(p.search(query) for p in patterns):
            hits.append(attr)
    # "What is the battery life ...?" opens with "what is the", which is not a request for the
    # product description: it only counts when nothing more specific was asked, or when the
    # user said describe / tell me about.
    if "description" in hits and len(hits) > 1 and not _DESCRIBE_RE.search(query):
        hits.remove("description")
    return hits


def detect_attributes(query: str) -> list[str]:
    """Public entry point kept for callers/tests that only need
    attributes: LLM-first via classify_query(), deterministic fallback
    on LLMUnavailable. Prefer classify_query() directly when intent is
    also needed -- it makes one LLM call for both instead of two."""
    attributes, _intent = classify_query(query)
    return attributes


# Numeric Step 2 attributes an exhaustive query can ask about by exact
# value ("Which products have 7 brushing modes?"), as opposed to the
# boolean attributes 3A.13 originally covered ("Which products have
# pressure sensing?"). Kept in sync by hand with
# src/rag/comparison_engine.py:NUMERIC_ATTRIBUTES -- not imported from
# there to avoid a 3A->3B dependency for what is just a small, stable
# set of Step 2 schema column names.
NUMERIC_EXHAUSTIVE_ATTRIBUTES = {"price", "mode_count", "battery_life_days"}

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def detect_exhaustive_condition_value(query: str, attribute: str):
    """The value a matching product's attribute must equal for an
    exhaustive "which products have <attribute>?" query (3A.13).

    Boolean attributes keep the original default: "which products have
    X" means X is present (True). Numeric attributes instead require a
    literal number in the query text itself (e.g. "7 brushing modes"
    -> mode_count == 7); returns None if no number is present, so the
    caller can tell "no numeric condition was actually stated" apart
    from "the condition is true" -- never silently falls back to
    comparing a number against True.
    """
    if attribute not in NUMERIC_EXHAUSTIVE_ATTRIBUTES:
        return True
    match = _NUMBER_RE.search(query)
    if not match:
        return None
    text = match.group(0)
    return int(text) if text.isdigit() else float(text)


# ---------------------------------------------------------------------------
# Intent classification (3A.13)
# ---------------------------------------------------------------------------

_EXHAUSTIVE_RE = re.compile(r"\bwhich (?:products?|models?|toothbrushes?)\b", re.I)
_COMPARISON_RE = re.compile(
    r"\bcompar\w*\b|\bvs\.?\b|\bversus\b"
    r"|\bdifference between\b"
    r"|\bwhich (?:is|one is|costs?|has)\b[^.?!]{0,40}\b(?:cheaper|more expensive|better|faster|lighter|heavier|less|more|fewer|higher|lower)\b"
    r"|\bby how much\b",
    re.I,
)
_COMPARISON_SPLIT_RE = re.compile(r"\bvs\.?\b|\bversus\b|\band\b|\bor\b|\bcompar\w*\b", re.I)


def _classify_intent_deterministic(query: str) -> Intent:
    """Regex fallback for classify_intent(): only reached when the LLM
    classifier is unavailable or fails -- see classify_query()."""
    if _EXHAUSTIVE_RE.search(query):
        return "exhaustive"
    if _COMPARISON_RE.search(query):
        return "comparison"
    return "normal"


def classify_intent(query: str, product_match: ProductMatch) -> Intent:
    """Public entry point kept for callers/tests that only need intent
    (product_match is unused -- intent never depended on it, even in
    the original regex version). LLM-first via classify_query(),
    deterministic fallback on LLMUnavailable. Prefer classify_query()
    directly when attributes are also needed -- one LLM call, not two."""
    _attributes, intent = classify_query(query)
    return intent


# ---------------------------------------------------------------------------
# LLM-first query classification (attributes + intent in one call).
# Both are open-vocabulary paraphrase problems (unlike product matching,
# a closed catalog lookup) -- a regex has to anticipate every phrasing
# ("how often should I replace the heads" vs "how long do heads last"
# vs "replacement schedule"); an LLM classifies from meaning instead.
# Falls back to the regex rules above only when the LLM is unavailable
# or fails (no API key, network error, bad JSON, ...) -- same
# LLMUnavailable discipline as every other LLM call in this app (3B's
# answer generation): never crashes the pipeline, never leaves the
# caller guessing which path ran.
# ---------------------------------------------------------------------------

_VALID_INTENTS: tuple[Intent, ...] = ("exhaustive", "comparison", "normal")


def _classification_system_prompt() -> str:
    attribute_lines = "\n".join(f"- {attr}" for attr in ATTRIBUTE_PATTERNS)
    return (
        "You classify a single user question about Philips Sonicare and "
        "Oral-B electric toothbrushes for a retrieval pipeline. Respond "
        'with ONLY a JSON object of the exact form {"attributes": '
        '["..."], "intent": "normal"} and nothing else.\n\n'
        '"attributes": every item from this exact list that the question '
        "is asking about, spelled exactly as shown (snake_case, lowercase); "
        "an empty list if the question does not ask about any of them. "
        "Pick attributes by what the question actually wants to know, not "
        "by shared words -- a question about how many replacement heads "
        "come in the box asks about included_items/brush_heads_included, "
        "NOT brush_head_replacement_interval (that is a different "
        "question: how often heads need replacing).\n"
        f"{attribute_lines}\n\n"
        '"intent": exactly one of:\n'
        '"exhaustive" -- a "which product(s)/model(s)" question expecting '
        "every matching product to be listed\n"
        '"comparison" -- the question compares two or more specifically '
        "named products against each other\n"
        '"normal" -- anything else, including a question about a single '
        "product"
    )


def _parse_classification(raw: dict) -> tuple[list[str], Intent]:
    attrs_raw = raw.get("attributes") if isinstance(raw, dict) else None
    # Hallucinated/unknown attribute names are dropped rather than passed
    # through -- every attribute name downstream must be a real
    # comparison_table.csv column (see ATTRIBUTE_PATTERNS's own docstring).
    attributes = [a for a in attrs_raw if a in ATTRIBUTE_PATTERNS] if isinstance(attrs_raw, list) else []
    intent_raw = raw.get("intent") if isinstance(raw, dict) else None
    intent: Intent = intent_raw if intent_raw in _VALID_INTENTS else "normal"
    return attributes, intent


def classify_query(
    query: str, llm_client: Optional[OpenAIStructuredClient] = None, use_llm: bool = True,
) -> tuple[list[str], Intent]:
    """The attributes (if any) and intent for `query`, LLM-first with a
    deterministic regex fallback. One LLM call covers both -- callers
    that need both should call this directly rather than detect_attributes()
    + classify_intent(), which would otherwise each make their own call.

    use_llm=False forces the deterministic path without even attempting
    a call -- for a caller that already asked for fully deterministic
    behavior (e.g. generation.py's --no-llm), so that request isn't
    half-honored by this stage still reaching out to the LLM."""
    if not use_llm:
        return _detect_attributes_deterministic(query), _classify_intent_deterministic(query)
    client = llm_client or OpenAIStructuredClient()
    try:
        raw = client.generate_json(_classification_system_prompt(), f"Question: {query}")
    except LLMUnavailable:
        return _detect_attributes_deterministic(query), _classify_intent_deterministic(query)
    return _parse_classification(raw)


def detect_comparison_products(query: str, products: list[ProductRecord]) -> list[ProductMatch]:
    """For a comparison-intent query ("Compare X and Y on Z"), a single
    detect_products() call can't tell two named products apart -- split
    on comparison delimiters (and/vs/versus/compar*) and resolve each
    segment independently, deduplicating by product_id so the same
    product named twice doesn't count twice."""
    segments = [s.strip() for s in _COMPARISON_SPLIT_RE.split(query) if s.strip()]
    matches: list[ProductMatch] = []
    seen_product_ids: set[str] = set()
    for segment in segments:
        match = detect_products(segment, products)
        if match.kind != "matched":
            continue
        product = match.candidates[0]
        if product.product_id in seen_product_ids:
            continue
        seen_product_ids.add(product.product_id)
        matches.append(match)
    return matches


def detect_comparison_not_in_corpus(query: str, products: list[ProductRecord]) -> bool:
    """True if a comparison query names a SPECIFIC product that is not
    in the corpus (3A.15 extended to comparisons) -- distinguishes "one
    of the two named products doesn't exist" (should refuse, same as a
    single-product not-in-corpus query) from "fewer than 2 products
    resolved because this wasn't really naming two specific models"
    (e.g. a vague "compare these" with nothing concrete to flag),
    which should still fall through to normal retrieval. Without this,
    a comparison naming one real and one nonexistent product silently
    answered about only the real one, with no indication the other
    product was never found."""
    segments = [s.strip() for s in _COMPARISON_SPLIT_RE.split(query) if s.strip()]
    return any(detect_products(segment, products).kind == "not_in_corpus" for segment in segments)
    return matches
