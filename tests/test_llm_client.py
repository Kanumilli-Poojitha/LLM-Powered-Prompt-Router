"""Unit tests for OpenAI-compatible client configuration behavior."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from app import llm_client
from app.llm_client import LLMClient


class FakeOpenAI:
    """Simple fake OpenAI client used to capture init and request args."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.last_create_kwargs: dict[str, object] = {}
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs: object) -> object:
        self.last_create_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_missing_api_key_raises_clear_error() -> None:
    os.environ["OPENAI_API_KEY"] = ""
    with pytest.raises(ValueError, match="Missing OPENAI_API_KEY"):
        LLMClient()


def test_empty_base_url_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    fake_instances: list[FakeOpenAI] = []

    def fake_factory(api_key: str, base_url: str | None = None) -> FakeOpenAI:
        instance = FakeOpenAI(api_key=api_key, base_url=base_url)
        fake_instances.append(instance)
        return instance

    monkeypatch.setattr(llm_client, "OpenAI", fake_factory)

    client = LLMClient()
    assert client.model == "llama-3.3-70b-versatile"
    assert fake_instances[0].base_url is None


def test_chat_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

    fake_instances: list[FakeOpenAI] = []

    def fake_factory(api_key: str, base_url: str | None = None) -> FakeOpenAI:
        instance = FakeOpenAI(api_key=api_key, base_url=base_url)
        fake_instances.append(instance)
        return instance

    monkeypatch.setattr(llm_client, "OpenAI", fake_factory)

    client = LLMClient()
    result = client.chat(system_prompt="s", user_message="u")

    assert result == "ok"
    assert fake_instances[0].base_url == "https://api.groq.com/openai/v1"
    assert fake_instances[0].last_create_kwargs["model"] == "llama-3.3-70b-versatile"
    assert fake_instances[0].last_create_kwargs["temperature"] == 0.2
    assert isinstance(fake_instances[0].last_create_kwargs["messages"], list)
