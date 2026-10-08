"""Tests for the 3C.2 shared, cached construction point
(src/app/engine.py). No business logic lives there -- these tests only
verify the caching contract the Streamlit app depends on: the
embedding model / FAISS index must not be rebuilt on every call."""
from __future__ import annotations

from src.app.engine import get_engine, get_llm_client
from src.rag.llm_provider import OpenAIStructuredClient
from src.rag.retrieve import RetrievalEngine


class TestGetEngineIsCached:
    def test_returns_a_retrieval_engine(self):
        engine = get_engine()
        assert isinstance(engine, RetrievalEngine)

    def test_repeated_calls_return_the_same_instance(self):
        """The core requirement: construction (which loads the
        embedding model) must happen once, not per call -- a Streamlit
        app re-executes this call on every user interaction."""
        first = get_engine()
        second = get_engine()
        assert first is second

    def test_cached_engine_can_actually_answer_a_query(self):
        """Not just "doesn't crash" -- the cached instance is a fully
        usable engine."""
        engine = get_engine()
        result = engine.retrieve("What is the price of the Sonicare Prestige 9900?")
        assert result.results or result.exhaustive_matches is not None or result.clarification_needed


class TestGetLlmClientIsCached:
    def test_returns_an_openai_structured_client(self):
        client = get_llm_client()
        assert isinstance(client, OpenAIStructuredClient)

    def test_repeated_calls_return_the_same_instance(self):
        first = get_llm_client()
        second = get_llm_client()
        assert first is second
