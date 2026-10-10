"""Pinned DeepEval judge configuration. Same style as
src/rag/llm_provider.py: a single model identifier, never chosen
dynamically, overridable only via an env var.

    DEEPEVAL_JUDGE_MODEL=<model id>   (defaults to DEFAULT_JUDGE_MODEL below)
    OPENAI_API_KEY=...                (already required for generation; reused here)

DeepEval's own anonymous usage telemetry is opted out at import time
(DEEPEVAL_TELEMETRY_OPT_OUT=1) -- this is unrelated to evaluation
results/data, which never leave this machine either way: nothing here
ever calls `deepeval login` or sets a Confident AI API key, so
`evaluate()`/metric `.measure()` calls run and print locally only. The
only network calls this module causes are judge-model requests to
OpenAI, same provider the app already trusts for generation.
"""
from __future__ import annotations

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

DEFAULT_JUDGE_MODEL = "gpt-4.1"


def judge_model() -> str:
    return os.environ.get("DEEPEVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def judge_available() -> bool:
    """DeepEval's default OpenAI-backed judge needs OPENAI_API_KEY in the
    real environment -- the same key src/rag/llm_provider.py reads."""
    return bool(os.environ.get("OPENAI_API_KEY"))
