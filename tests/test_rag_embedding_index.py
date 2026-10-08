"""Tests for the FAISS-backed semantic index. Uses a fake, deterministic
encoder (monkeypatched in) so these stay fast and don't require loading
the real ~90MB sentence-transformers model -- the real model is only
exercised by the slower integration tests (test_rag_retrieve_integration.py)
via the shared `real_engine` session fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.rag.embedding_index import EmbeddingIndex
from src.rag.models import RetrievalDocument


def _doc(doc_id: str, text: str) -> RetrievalDocument:
    return RetrievalDocument(
        document_id=doc_id, evidence_id=doc_id, product_id="p1", brand="Oral-B",
        product_name="p1", attribute="pressure_sensing", value=True, status="stated",
        evidence_text=text, source_url="https://oralb.com/x", source_section="s",
        source_file="x.html", retrieval_text=text,
    )


class _FakeModel:
    """Deterministic bag-of-words-ish encoder: embeds a string as a
    normalized vector derived from its character codes, so identical
    text always gets an identical vector and different text differs."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        dim = 16
        vectors = []
        for t in texts:
            v = np.zeros(dim, dtype="float32")
            for i, ch in enumerate(t):
                v[ord(ch) % dim] += 1.0
            norm = np.linalg.norm(v)
            if normalize_embeddings and norm > 0:
                v = v / norm
            vectors.append(v)
        return np.array(vectors, dtype="float32")


@pytest.fixture
def fake_index(monkeypatch) -> EmbeddingIndex:
    index = EmbeddingIndex()
    monkeypatch.setattr(index, "_get_model", lambda: _FakeModel())
    return index


class TestEmbeddingIndexBuildSearch:
    def test_search_returns_results_sorted_by_score(self, fake_index):
        docs = [_doc("d1", "Smart Pressure Sensor"), _doc("d2", "7 Smart Brushing Modes")]
        fake_index.build(docs)
        results = fake_index.search("Smart Pressure Sensor", top_k=2)
        assert results
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_identical_text_scores_highest(self, fake_index):
        docs = [_doc("d1", "Smart Pressure Sensor"), _doc("d2", "Magnetic Charger")]
        fake_index.build(docs)
        results = fake_index.search("Smart Pressure Sensor", top_k=2)
        assert results[0][0].document_id == "d1"

    def test_empty_corpus_search_returns_nothing(self, fake_index):
        fake_index.build([])
        assert fake_index.search("anything", top_k=5) == []

    def test_empty_query_returns_nothing(self, fake_index):
        fake_index.build([_doc("d1", "x")])
        assert fake_index.search("", top_k=5) == []


class TestEmbeddingIndexSaveLoad:
    def test_round_trip_preserves_search_behavior(self, fake_index, tmp_path, monkeypatch):
        docs = [_doc("d1", "Smart Pressure Sensor"), _doc("d2", "Magnetic Charger")]
        fake_index.build(docs)
        index_path = tmp_path / "faiss.index"
        metadata_path = tmp_path / "metadata.json"
        fake_index.save(index_path, metadata_path)

        loaded = EmbeddingIndex.load(index_path, metadata_path, docs)
        monkeypatch.setattr(loaded, "_get_model", lambda: _FakeModel())
        results = loaded.search("Smart Pressure Sensor", top_k=2)
        assert results[0][0].document_id == "d1"

    def test_load_raises_clearly_if_metadata_references_missing_documents(self, fake_index, tmp_path):
        docs = [_doc("d1", "x"), _doc("d2", "y")]
        fake_index.build(docs)
        index_path = tmp_path / "faiss.index"
        metadata_path = tmp_path / "metadata.json"
        fake_index.save(index_path, metadata_path)

        with pytest.raises(ValueError):
            EmbeddingIndex.load(index_path, metadata_path, [docs[0]])  # d2 missing
