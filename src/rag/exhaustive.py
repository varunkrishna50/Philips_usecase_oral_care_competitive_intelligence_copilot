"""Exhaustive "Which products have <attribute>?" queries (3A.13).

This path is deliberately NOT top-k retrieval: top-k over a hybrid/BM25/
semantic score can never guarantee every true match surfaces (a product
whose evidence happens to score lower is just as true a match as one
that scores higher). Instead this scans the COMPLETE product corpus --
the Step 2 source of truth -- and returns every product whose attribute
value satisfies the condition, each with its own supporting evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import ProductRecord, RetrievalDocument


@dataclass
class ExhaustiveMatch:
    product: ProductRecord
    evidence: list[RetrievalDocument]


def exhaustive_query(
    attribute: str,
    product_corpus: list[ProductRecord],
    retrieval_documents: list[RetrievalDocument],
    condition_value: object = True,
) -> list[ExhaustiveMatch]:
    """Every product in the corpus where product.attributes[attribute]
    == condition_value (default True, for the boolean attributes named
    in 3A.13: pressure_sensing, app_connectivity, bluetooth_connectivity,
    timer), paired with that product's own evidence for the attribute.
    """
    docs_by_product_attr: dict[tuple[str, str], list[RetrievalDocument]] = {}
    for doc in retrieval_documents:
        docs_by_product_attr.setdefault((doc.product_id, doc.attribute), []).append(doc)

    matches: list[ExhaustiveMatch] = []
    for product in product_corpus:
        if product.attributes.get(attribute) == condition_value:
            evidence = docs_by_product_attr.get((product.product_id, attribute), [])
            matches.append(ExhaustiveMatch(product=product, evidence=evidence))
    return matches
