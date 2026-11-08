from pathlib import Path

import pytest
from pydantic import ValidationError

from wanderlust.config import LLMProvider, MissingAPIKeyError, Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Keep the developer's real environment and .env file out of the tests."""
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "WANDERLUST_LLM_PROVIDER",
        "WANDERLUST_LLM_MODEL",
        "WANDERLUST_WIKIVOYAGE_LANGUAGE",
        "WANDERLUST_DATA_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_defaults():
    settings = make_settings()
    assert settings.llm_provider is LLMProvider.OPENAI
    assert settings.llm_model is None
    assert settings.wikivoyage_language == "en"
    assert settings.data_dir == Path(".wanderlust")


def test_reads_prefixed_env_vars(monkeypatch):
    monkeypatch.setenv("WANDERLUST_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("WANDERLUST_WIKIVOYAGE_LANGUAGE", "de")
    settings = make_settings()
    assert settings.llm_provider is LLMProvider.ANTHROPIC
    assert settings.wikivoyage_language == "de"


def test_api_key_uses_standard_env_names(monkeypatch):
    monkeypatch.setenv("WANDERLUST_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    assert make_settings().api_key() == "test-anthropic-key"


def test_missing_key_raises_helpful_error():
    settings = make_settings()
    assert not settings.has_api_key()
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        settings.api_key()


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert not make_settings().has_api_key()


def test_api_key_is_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")
    assert "test-secret-value" not in repr(make_settings())


def test_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("WANDERLUST_LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValidationError):
        make_settings()


def test_rejects_invalid_language_code(monkeypatch):
    monkeypatch.setenv("WANDERLUST_WIKIVOYAGE_LANGUAGE", "English")
    with pytest.raises(ValidationError):
        make_settings()
