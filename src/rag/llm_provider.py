"""LLM provider (3B.2/3B.3/3B.4): a single pinned OpenAI model/temperature,
configured only through environment variables -- never chosen or
substituted dynamically at runtime.

    LLM_PROVIDER=openai        (only "openai" is implemented)
    LLM_MODEL=<model id>       (defaults to DEFAULT_MODEL below)
    OPENAI_API_KEY=...

Every failure mode (no key, package missing, network/API error,
malformed JSON back from the model) raises the SAME LLMUnavailable
exception, which callers (generation.py) must catch and answer from the
deterministic fallback (3B.3) instead of crashing the application.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Loads a local .env (gitignored, never committed -- see .env.example for
# the documented placeholder shape) into os.environ exactly once per
# process. override=False (the library default, stated explicitly here)
# means a variable already set in the real environment always wins over
# whatever .env says -- .env only fills gaps, never shadows a real value.
load_dotenv(override=False)

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0


class LLMUnavailable(Exception):
    """No OPENAI_API_KEY, the openai package is missing, the request
    failed, or the model didn't return valid JSON -- in every case the
    caller must fall back to the deterministic answer, never crash."""


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    api_key: Optional[str]

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            temperature=TEMPERATURE,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    @property
    def available(self) -> bool:
        return self.provider == "openai" and bool(self.api_key)


class OpenAIStructuredClient:
    """Calls the pinned model with temperature=0 and asks for a JSON
    object back; returns it parsed. No LLM call anywhere else in 3b
    bypasses this one chokepoint."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        # Introspection only -- never read by generate_json itself, never
        # changes what's sent. Lets callers (verify_llm.py) report
        # tokens/latency per call and confirm whether a call happened at
        # all (e.g. to prove comparison answers never invoke the LLM).
        self.call_count = 0
        self.last_usage: Optional[dict] = None
        self.last_latency_seconds: Optional[float] = None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        if not self.config.available:
            raise LLMUnavailable(
                f"LLM unavailable: provider={self.config.provider!r}, "
                f"api_key_set={bool(self.config.api_key)} -- using deterministic fallback."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailable(f"openai package not installed: {exc}") from exc

        client = OpenAI(api_key=self.config.api_key)
        self.call_count += 1
        start = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - any API/network failure degrades to fallback
            raise LLMUnavailable(f"OpenAI request failed ({self.config.model}): {exc}") from exc
        finally:
            self.last_latency_seconds = time.monotonic() - start

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMUnavailable("OpenAI response had no content")
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError) as exc:
            raise LLMUnavailable(f"LLM did not return valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMUnavailable("LLM JSON response was not an object")
        return parsed
