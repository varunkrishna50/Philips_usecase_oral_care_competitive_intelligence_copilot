"""Step 3C.2: the one shared, cached construction point for the
retrieval engine and LLM client that a long-lived process (Streamlit)
needs and a one-shot CLI process does not.

No business logic lives here. `answer_query()` (src/rag/generation.py)
remains the entire application/service API -- this module only builds
and caches the two objects it needs:

- RetrievalEngine: loading the embedding model + FAISS index + BM25
  index takes ~8-15s. A one-shot CLI process (src/rag/pipeline.py,
  src/rag/retrieve.py) pays that cost once per invocation already,
  which is fine. Streamlit re-executes the whole script on every user
  interaction, so without caching it would pay that cost on every
  question -- st.cache_resource makes construction happen exactly once
  per server process instead.
- OpenAIStructuredClient: cheap to construct (holds only config +
  introspection fields), but reused here anyway per the caching
  requirement, rather than letting answer_query() construct a fresh
  one internally on every call.

Importable without Streamlit installed: outside of a running
Streamlit script (e.g. a plain Python process, or a test), the
decorator degrades to a plain in-process cache using functools.cache,
so get_engine()/get_llm_client() still only construct their object
once per process either way.
"""
from __future__ import annotations

from src.rag.llm_provider import OpenAIStructuredClient
from src.rag.retrieve import RetrievalEngine

try:
    import streamlit as st

    _cache_resource = st.cache_resource
except ImportError:  # pragma: no cover - exercised only when streamlit isn't installed
    import functools

    def _cache_resource(*args, **kwargs):
        """Mimics st.cache_resource's dual use (bare @_cache_resource,
        or @_cache_resource(show_spinner=...)) with a plain
        functools.cache, ignoring any streamlit-only kwargs."""
        if args and callable(args[0]):
            return functools.cache(args[0])

        def decorator(func):
            return functools.cache(func)

        return decorator


@_cache_resource(show_spinner="Loading retrieval engine and embedding model (first run only)...")
def get_engine() -> RetrievalEngine:
    """Loads the persisted FAISS index if present, else builds the
    index in memory -- identical fallback behavior to the existing
    CLI entry points (src/rag/pipeline.py, src/rag/retrieve.py), which
    both already call RetrievalEngine.load() directly."""
    return RetrievalEngine.load()


@_cache_resource
def get_llm_client() -> OpenAIStructuredClient:
    """A single reused client per process, constructed from whatever
    LLM_PROVIDER/LLM_MODEL/OPENAI_API_KEY is configured (via .env or
    the real environment) -- see src/rag/llm_provider.py. Safe to call
    unconditionally even with no key configured: answer_query() already
    handles an unavailable client via its existing deterministic
    fallback (3B.3), so no availability check belongs here."""
    return OpenAIStructuredClient()
