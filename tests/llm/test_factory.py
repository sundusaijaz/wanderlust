import pytest

from wanderlust.config import LLMProvider as ProviderName
from wanderlust.config import MissingAPIKeyError, Settings
from wanderlust.llm.anthropic_provider import AnthropicProvider
from wanderlust.llm.factory import build_provider, default_model
from wanderlust.llm.openai_provider import OpenAIProvider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "WANDERLUST_LLM_PROVIDER",
        "WANDERLUST_LLM_MODEL",
        "WANDERLUST_LLM_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_default_models():
    assert default_model(ProviderName.ANTHROPIC) == "claude-opus-5-5"
    assert default_model(ProviderName.OPENAI) == "gpt-5-mini"


def test_builds_anthropic_with_medium_effort_by_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = build_provider(settings(llm_provider="anthropic"))
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-opus-5-5"
    assert provider.effort == "medium"
    assert provider.fallbacks


def test_anthropic_model_and_effort_overrides(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = build_provider(
        settings(llm_provider="anthropic", llm_model="claude-haiku-5-5", llm_effort="low")
    )
    assert provider.model == "claude-haiku-5-5"
    assert provider.effort == "low"
    assert not provider.fallbacks  # Haiku has no server-side fallback


def test_builds_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = build_provider(settings())
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-5-mini"
    assert provider.reasoning_effort is None


@pytest.mark.parametrize(("effort", "expected"), [("low", "low"), ("max", "high")])
def test_openai_effort_mapping(monkeypatch, effort, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert build_provider(settings(llm_effort=effort)).reasoning_effort == expected


def test_missing_key_is_reported():
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        build_provider(settings(llm_provider="anthropic"))
