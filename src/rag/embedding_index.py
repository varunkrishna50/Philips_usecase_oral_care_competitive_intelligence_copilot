"""Semantic retrieval: sentence-transformers embeddings + a FAISS index.

The model is lazy-loaded (only on first build()/search() call) so
importing this module, or constructing an EmbeddingIndex, never pays the
~15s model-load cost or requires a network call unless semantic search
is actually used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from .config import EMBEDDING_MODEL_NAME
from .models import RetrievalDocument


class EmbeddingIndex:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self.index: Optional["faiss.Index"] = None
        self.documents: list[RetrievalDocument] = []

    def _get_model(self):
        if self._model is None:
            import os

            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                # A transient network error (e.g. a reset connection
                # during HF Hub's cache-freshness check) shouldn't fail
                # a load when the model is already cached locally from a
                # prior run -- retry once, forcing the local cache only.
                prior = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    self._model = SentenceTransformer(self.model_name)
                finally:
                    if prior is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = prior
        return self._model

    def build(self, documents: list[RetrievalDocument]) -> None:
        self.documents = documents
        if not documents:
            self.index = faiss.IndexFlatIP(1)
            return
        model = self._get_model()
        texts = [d.retrieval_text for d in documents]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index

    def search(self, query: str, top_k: int = 10) -> list[tuple[RetrievalDocument, float]]:
        if self.index is None or not self.documents or not query.strip():
            return []
        model = self._get_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.documents[int(idx)], float(score)))
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata = {
            "model_name": self.model_name,
            "dim": int(self.index.d),
            "count": len(self.documents),
            "document_ids": [d.document_id for d in self.documents],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path, documents: list[RetrievalDocument]) -> "EmbeddingIndex":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        obj = cls(model_name=metadata["model_name"])
        obj.index = faiss.read_index(str(index_path))
        by_id = {d.document_id: d for d in documents}
        missing = [doc_id for doc_id in metadata["document_ids"] if doc_id not in by_id]
        if missing:
            raise ValueError(
                f"FAISS metadata references {len(missing)} document id(s) not present in the "
                f"supplied retrieval documents (e.g. {missing[:3]}) -- rebuild the index with "
                "scripts/build_index.py against the current corpus."
            )
        obj.documents = [by_id[doc_id] for doc_id in metadata["document_ids"]]
        return obj
