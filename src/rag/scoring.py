"""Hybrid score combination (3A.7).

HybridScore = alpha * BM25_norm + (1 - alpha) * Semantic_norm

Both score lists are min-max normalized to [0, 1] independently before
combining, since BM25's raw scores (unbounded) and FAISS inner-product
scores (bounded, since both vectors are L2-normalized -> cosine
similarity) live on different scales. Callers should retrieve over the
FULL corpus from both BM25 and the embedding index (not just each
method's own top-k) so a document either method ranked low still gets a
meaningful combined score instead of defaulting to 0 for "not in this
method's top-k" -- see retrieve.py.
"""
from __future__ import annotations

from .config import DEFAULT_ALPHA
from .models import RetrievalDocument, ScoredDocument


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: (1.0 if v > 0 else 0.0) for k, v in scores.items()}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def combine_hybrid(
    bm25_results: list[tuple[RetrievalDocument, float]],
    semantic_results: list[tuple[RetrievalDocument, float]],
    alpha: float = DEFAULT_ALPHA,
) -> list[ScoredDocument]:
    bm25_by_id = {doc.document_id: score for doc, score in bm25_results}
    sem_by_id = {doc.document_id: score for doc, score in semantic_results}
    docs_by_id = {doc.document_id: doc for doc, _ in bm25_results}
    docs_by_id.update({doc.document_id: doc for doc, _ in semantic_results})

    bm25_norm = min_max_normalize(bm25_by_id)
    sem_norm = min_max_normalize(sem_by_id)

    scored: list[ScoredDocument] = []
    for doc_id, doc in docs_by_id.items():
        b_raw, b_norm = bm25_by_id.get(doc_id, 0.0), bm25_norm.get(doc_id, 0.0)
        s_raw, s_norm = sem_by_id.get(doc_id, 0.0), sem_norm.get(doc_id, 0.0)
        hybrid = alpha * b_norm + (1 - alpha) * s_norm
        scored.append(
            ScoredDocument(
                document=doc,
                bm25_score=b_raw, bm25_norm=b_norm,
                semantic_score=s_raw, semantic_norm=s_norm,
                hybrid_score=hybrid,
            )
        )
    scored.sort(key=lambda sd: sd.hybrid_score, reverse=True)
    return scored
