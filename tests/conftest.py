import os
import sys
from pathlib import Path

import pytest

# faiss and torch (pulled in by sentence-transformers) each bundle their
# own OpenMP runtime; loading both in one process aborts on macOS with
# "OMP: Error #15" unless this is set before either is imported. This is
# a known library-interaction issue, not a correctness concern -- both
# libraries' math still runs single-threaded-safe for our tiny corpus.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
# Step 3 (src/rag) is addressed as `src.rag.*` (see src/__init__.py), which
# needs the project ROOT itself on sys.path, not just src/.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_real_llm_key(monkeypatch):
    """Keeps every real, billed OpenAI API call out of the automated
    test suite (3B checkpoint requirement), regardless of whether a
    local .env configures a real OPENAI_API_KEY for manual use (e.g.
    scripts/verify_llm.py). Deleting it from os.environ for the
    duration of each test forces LLMConfig.from_env().available to be
    False everywhere in pytest -- confirmed necessary: without this,
    a test that calls answer_query() without an explicit --no-llm or a
    fake client silently made a real network call whenever .env was
    present locally."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(scope="session")
def real_engine():
    """The real RetrievalEngine built from the actual Step 2 output,
    shared across all tests that need it so the embedding model only
    loads once per test session (~8-15s) instead of once per test."""
    from src.rag.retrieve import RetrievalEngine

    return RetrievalEngine.build()
