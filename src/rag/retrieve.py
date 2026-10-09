"""Retrieval orchestration (3a) + CLI debug tool.

    python -m src.rag.retrieve --query "..." [--debug] [--top-k 10] [--alpha 0.5]

Ties together: query understanding -> ambiguous/not-in-corpus short
circuits -> exhaustive-query path -> comparison path -> standard hybrid
retrieval (BM25 + semantic over the FULL corpus) -> rerank -> hard
evidence filter. No LLM anywhere in this module.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .bm25_index import BM25Index
from .config import DEFAULT_ALPHA, DEFAULT_TOP_K, FAISS_INDEX_PATH, FAISS_METADATA_PATH
from .corpus import build_retrieval_documents, load_product_corpus
from .embedding_index import EmbeddingIndex
from .attribute_status import expand_attributes
from .evidence_filter import FilterResult, filter_evidence
from .exhaustive import ExhaustiveMatch, exhaustive_query
from .models import ProductRecord, RetrievalDocument, ScoredDocument
from .query_understanding import (
    ProductMatch,
    classify_intent,
    detect_attributes,
    detect_comparison_products,
    detect_exhaustive_condition_value,
    detect_products,
)
from .rerank import rerank
from .scoring import combine_hybrid


# user-facing text when the named product is not in the data
NOT_IN_CORPUS_MESSAGE = "I couldn't find that product on the Philips US or Oral-B US pages I have."


@dataclass
class RetrievalResult:
    query: str
    intent: str
    product_match: ProductMatch
    attributes: list[str]

    results: list[ScoredDocument] = field(default_factory=list)
    rejected: list[tuple[ScoredDocument, str]] = field(default_factory=list)

    clarification_needed: bool = False
    clarification_message: Optional[str] = None
    not_in_corpus: bool = False
    not_in_corpus_message: Optional[str] = None

    exhaustive_matches: Optional[list[ExhaustiveMatch]] = None
    comparison_results: Optional[dict[str, list[ScoredDocument]]] = None

    # Raw intermediate candidates, kept only for --debug introspection.
    bm25_candidates: list[tuple[RetrievalDocument, float]] = field(default_factory=list)
    semantic_candidates: list[tuple[RetrievalDocument, float]] = field(default_factory=list)
    hybrid_ranking: list[ScoredDocument] = field(default_factory=list)


class RetrievalEngine:
    def __init__(
        self,
        product_corpus: list[ProductRecord],
        retrieval_documents: list[RetrievalDocument],
        bm25_index: BM25Index,
        embedding_index: EmbeddingIndex,
    ):
        self.product_corpus = product_corpus
        self.retrieval_documents = retrieval_documents
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index

    @classmethod
    def build(cls) -> "RetrievalEngine":
        """Builds everything in memory (no persisted FAISS index
        required) -- used by tests and by scripts/build_index.py itself.
        Loading the embedding model happens lazily on first search."""
        product_corpus = load_product_corpus()
        retrieval_documents = build_retrieval_documents(product_corpus=product_corpus)
        bm25_index = BM25Index(retrieval_documents)
        embedding_index = EmbeddingIndex()
        embedding_index.build(retrieval_documents)
        return cls(product_corpus, retrieval_documents, bm25_index, embedding_index)

    @classmethod
    def load(cls, index_path: Path = FAISS_INDEX_PATH, metadata_path: Path = FAISS_METADATA_PATH) -> "RetrievalEngine":
        """Loads the persisted FAISS index (built by scripts/build_index.py)
        instead of re-encoding the whole corpus -- the normal path for the
        CLI/evaluation scripts. Falls back to an in-memory build if the
        persisted index is missing, so a fresh checkout still works."""
        product_corpus = load_product_corpus()
        retrieval_documents = build_retrieval_documents(product_corpus=product_corpus)
        bm25_index = BM25Index(retrieval_documents)
        if index_path.exists() and metadata_path.exists():
            embedding_index = EmbeddingIndex.load(index_path, metadata_path, retrieval_documents)
        else:
            embedding_index = EmbeddingIndex()
            embedding_index.build(retrieval_documents)
        return cls(product_corpus, retrieval_documents, bm25_index, embedding_index)

    def _hybrid_rerank_filter(
        self, query: str, product_match: ProductMatch, attributes: list[str], top_k: int, alpha: float
    ) -> tuple[list[ScoredDocument], list[tuple[ScoredDocument, str]], list[tuple], list[tuple], list[ScoredDocument]]:
        n = len(self.retrieval_documents)
        bm25_candidates = self.bm25_index.search(query, top_k=n)
        semantic_candidates = self.embedding_index.search(query, top_k=n)
        hybrid_ranking = combine_hybrid(bm25_candidates, semantic_candidates, alpha=alpha)
        # the asked attributes plus the detail attributes that explain them
        # ("pressure_sensing" -> "pressure_sensing_details"); result.attributes stays the asked list
        target_attrs = expand_attributes(attributes)
        reranked = rerank(list(hybrid_ranking), product_match, target_attrs)
        filter_result: FilterResult = filter_evidence(reranked, product_match, target_attrs)
        return filter_result.accepted[:top_k], filter_result.rejected, bm25_candidates, semantic_candidates, hybrid_ranking

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
    ) -> RetrievalResult:
        product_match = detect_products(query, self.product_corpus)
        attributes = detect_attributes(query)
        intent = classify_intent(query, product_match)

        result = RetrievalResult(query=query, intent=intent, product_match=product_match, attributes=attributes)

        # 3A.14 -- ambiguous product reference: clarify, don't guess.
        if product_match.kind == "ambiguous":
            names = ", ".join(sorted({p.product_name for p in product_match.candidates}))
            brand = product_match.candidates[0].brand
            result.clarification_needed = True
            result.clarification_message = (
                f"I found multiple {brand} products matching this query ({len(product_match.candidates)} "
                f"candidates: {names}). Please specify further (e.g. the exact series/model number)."
            )
            return result

        # 3A.15 -- product named but not in the corpus: say so, don't substitute.
        if product_match.kind == "not_in_corpus":
            result.not_in_corpus = True
            result.not_in_corpus_message = NOT_IN_CORPUS_MESSAGE
            return result

        # 3A.13 -- exhaustive "which products have X" bypasses top-k entirely.
        # The condition each matching product must satisfy depends on
        # the attribute's type: boolean attributes default to True
        # ("which products have pressure sensing?"); numeric attributes
        # require a literal number stated in the query itself ("which
        # products have 7 brushing modes?" -> mode_count == 7). A
        # numeric attribute named with no number in the query has no
        # well-defined condition and is skipped, not silently compared
        # against True.
        if intent == "exhaustive":
            matches: list[ExhaustiveMatch] = []
            for attribute in attributes:
                condition_value = detect_exhaustive_condition_value(query, attribute)
                if condition_value is None:
                    continue
                matches.extend(exhaustive_query(attribute, self.product_corpus, self.retrieval_documents, condition_value))
            result.exhaustive_matches = matches
            return result

        # Comparison: resolve each named product independently and run
        # the standard pipeline scoped to each one.
        if intent == "comparison":
            comparison_matches = detect_comparison_products(query, self.product_corpus)
            if len(comparison_matches) >= 2:
                comparison_results: dict[str, list[ScoredDocument]] = {}
                for pm in comparison_matches:
                    accepted, rejected, bm25_c, sem_c, hybrid_r = self._hybrid_rerank_filter(
                        query, pm, attributes, top_k, alpha
                    )
                    comparison_results[pm.candidates[0].product_id] = accepted
                    result.rejected.extend(rejected)
                result.comparison_results = comparison_results
                result.results = [doc for docs in comparison_results.values() for doc in docs]
                return result
            # Fewer than 2 products resolved -- fall through to normal retrieval.

        accepted, rejected, bm25_c, sem_c, hybrid_r = self._hybrid_rerank_filter(
            query, product_match, attributes, top_k, alpha
        )
        result.results = accepted
        result.rejected = rejected
        result.bm25_candidates = bm25_c[:top_k]
        result.semantic_candidates = sem_c[:top_k]
        result.hybrid_ranking = hybrid_r[:top_k]
        return result


def _format_doc_line(doc: RetrievalDocument, score: float) -> str:
    return f"  [{score:+.4f}] {doc.product_id} :: {doc.attribute} :: {doc.evidence_text[:80]!r}"


def print_debug(result: RetrievalResult) -> None:
    print("Query interpretation:")
    print(f"  products: {[p.product_id for p in result.product_match.candidates]} (kind={result.product_match.kind})")
    print(f"  attributes: {result.attributes}")
    print(f"  intent: {result.intent}")
    print()

    if result.clarification_needed:
        print("CLARIFICATION NEEDED:", result.clarification_message)
        return
    if result.not_in_corpus:
        print("NOT IN CORPUS:", result.not_in_corpus_message)
        return
    if result.exhaustive_matches is not None:
        print(f"Exhaustive matches: {len(result.exhaustive_matches)}")
        for m in result.exhaustive_matches:
            print(f"  {m.product.product_id} (evidence: {len(m.evidence)})")
        return
    if result.comparison_results is not None:
        print("Comparison results:")
        for product_id, docs in result.comparison_results.items():
            print(f"  {product_id}: {len(docs)} accepted evidence docs")
        return

    print("BM25 candidates:")
    for doc, score in result.bm25_candidates[:10]:
        print(_format_doc_line(doc, score))
    print()
    print("Semantic candidates:")
    for doc, score in result.semantic_candidates[:10]:
        print(_format_doc_line(doc, score))
    print()
    print("Hybrid ranking:")
    for sd in result.hybrid_ranking[:10]:
        print(_format_doc_line(sd.document, sd.hybrid_score))
    print()
    print("Reranked + filtered results (accepted):")
    for sd in result.results:
        print(f"  tier={sd.rerank_tier} [{sd.hybrid_score:+.4f}] {sd.document.product_id} :: {sd.document.attribute} :: {sd.document.evidence_text[:80]!r}")
    print()
    print(f"Evidence rejected: {len(result.rejected)}")
    for sd, reason in result.rejected[:10]:
        print(f"  {sd.document.product_id} :: {sd.document.attribute} -- {reason}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    engine = RetrievalEngine.load()
    result = engine.retrieve(args.query, top_k=args.top_k, alpha=args.alpha)

    if args.debug:
        print_debug(result)
    else:
        if result.clarification_needed:
            print(result.clarification_message)
        elif result.not_in_corpus:
            print(result.not_in_corpus_message)
        elif result.exhaustive_matches is not None:
            for m in result.exhaustive_matches:
                print(f"{m.product.product_id}: {m.product.product_name}")
        elif result.comparison_results is not None:
            for product_id, docs in result.comparison_results.items():
                print(f"{product_id}: {len(docs)} evidence docs")
        else:
            for sd in result.results:
                print(_format_doc_line(sd.document, sd.hybrid_score))
    return 0


if __name__ == "__main__":
    sys.exit(main())
