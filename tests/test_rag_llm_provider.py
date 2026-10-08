"""Unit tests for the pinned LLM provider (3B.2/3B.3/3B.4): config comes
only from env vars, every failure mode degrades to the same
LLMUnavailable (never a raw exception), and the model/provider/
temperature are never chosen dynamically."""
from __future__ import annotations

import json

import pytest

from src.rag.llm_provider import DEFAULT_MODEL, DEFAULT_PROVIDER, TEMPERATURE, LLMConfig, LLMUnavailable, OpenAIStructuredClient


class TestLLMConfig:
    def test_defaults_when_no_env_vars_set(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = LLMConfig.from_env()
        assert config.provider == DEFAULT_PROVIDER == "openai"
        assert config.model == DEFAULT_MODEL
        assert config.temperature == TEMPERATURE == 0
        assert config.api_key is None
        assert not config.available

    def test_available_requires_both_openai_provider_and_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert LLMConfig.from_env().available

    def test_not_available_without_key_even_if_provider_set(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert not LLMConfig.from_env().available

    def test_temperature_is_always_zero_regardless_of_env(self, monkeypatch):
        """Temperature is pinned in code, not configurable -- no env var
        for it exists at all."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert LLMConfig.from_env().temperature == 0

    def test_model_is_overridable_via_env_but_defaults_to_pinned_value(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        assert LLMConfig.from_env().model == DEFAULT_MODEL
        monkeypatch.setenv("LLM_MODEL", "gpt-custom")
        assert LLMConfig.from_env().model == "gpt-custom"


class TestOpenAIStructuredClientNoKey:
    def test_raises_llm_unavailable_without_a_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = OpenAIStructuredClient(LLMConfig.from_env())
        with pytest.raises(LLMUnavailable):
            client.generate_json("system", "user")


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc:
            raise self._exc
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, completions, api_key=None):
        self.chat = _FakeChat(completions)
        self.api_key = api_key


class TestOpenAIStructuredClientWithKey:
    def _config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        return LLMConfig.from_env()

    def test_successful_json_response_is_parsed(self, monkeypatch):
        config = self._config(monkeypatch)
        completions = _FakeCompletions(content=json.dumps({"segments": [{"text": "x", "citation_ids": ["E1"]}]}))
        monkeypatch.setattr("openai.OpenAI", lambda api_key: _FakeOpenAIClient(completions, api_key))
        client = OpenAIStructuredClient(config)
        result = client.generate_json("sys", "user")
        assert result == {"segments": [{"text": "x", "citation_ids": ["E1"]}]}

    def test_pinned_model_and_zero_temperature_are_sent(self, monkeypatch):
        config = self._config(monkeypatch)
        completions = _FakeCompletions(content=json.dumps({"segments": []}))
        monkeypatch.setattr("openai.OpenAI", lambda api_key: _FakeOpenAIClient(completions, api_key))
        client = OpenAIStructuredClient(config)
        client.generate_json("sys", "user")
        assert completions.last_kwargs["model"] == DEFAULT_MODEL
        assert completions.last_kwargs["temperature"] == 0
        assert completions.last_kwargs["response_format"] == {"type": "json_object"}

    def test_network_error_degrades_to_llm_unavailable(self, monkeypatch):
        config = self._config(monkeypatch)
        completions = _FakeCompletions(exc=RuntimeError("connection reset"))
        monkeypatch.setattr("openai.OpenAI", lambda api_key: _FakeOpenAIClient(completions, api_key))
        client = OpenAIStructuredClient(config)
        with pytest.raises(LLMUnavailable):
            client.generate_json("sys", "user")

    def test_malformed_json_degrades_to_llm_unavailable(self, monkeypatch):
        config = self._config(monkeypatch)
        completions = _FakeCompletions(content="not json{")
        monkeypatch.setattr("openai.OpenAI", lambda api_key: _FakeOpenAIClient(completions, api_key))
        client = OpenAIStructuredClient(config)
        with pytest.raises(LLMUnavailable):
            client.generate_json("sys", "user")

    def test_non_object_json_degrades_to_llm_unavailable(self, monkeypatch):
        config = self._config(monkeypatch)
        completions = _FakeCompletions(content=json.dumps(["not", "an", "object"]))
        monkeypatch.setattr("openai.OpenAI", lambda api_key: _FakeOpenAIClient(completions, api_key))
        client = OpenAIStructuredClient(config)
        with pytest.raises(LLMUnavailable):
            client.generate_json("sys", "user")

    def test_empty_content_degrades_to_llm_unavailable(self, monkeypatch):
        config = self._config(monkeypatch)
        completions = _FakeCompletions(content="")
        monkeypatch.setattr("openai.OpenAI", lambda api_key: _FakeOpenAIClient(completions, api_key))
        client = OpenAIStructuredClient(config)
        with pytest.raises(LLMUnavailable):
            client.generate_json("sys", "user")
