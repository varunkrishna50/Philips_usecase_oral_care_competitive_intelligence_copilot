"""BM25 lexical retrieval over the evidence corpus.

Cheap to rebuild (no persistence needed): tokenizing ~500 short
documents and building the rank_bm25 index takes a few milliseconds, so
callers just construct a BM25Index from the retrieval documents each
run rather than serializing BM25 state to disk.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from .models import RetrievalDocument
from .query_understanding import normalize_tokens


class BM25Index:
    def __init__(self, documents: list[RetrievalDocument]):
        self.documents = documents
        self._tokenized = [normalize_tokens(d.retrieval_text) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized) if documents else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[RetrievalDocument, float]]:
        """Returns up to top_k (document, raw_bm25_score) pairs, sorted
        by score descending. Exact product names/IDs/SKUs/model numbers
        and attribute/feature terminology all work as plain BM25 terms
        since retrieval_text embeds the product name and attribute name
        verbatim (see corpus.py's _retrieval_text)."""
        if self._bm25 is None or not query.strip():
            return []
        tokens = normalize_tokens(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self.documents, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked[:top_k] if score > 0]
