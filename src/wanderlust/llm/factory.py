"""Create the configured language model provider."""

from __future__ import annotations

from wanderlust.config import Effort, Settings
from wanderlust.config import LLMProvider as ProviderName
from wanderlust.llm import anthropic_provider, openai_provider
from wanderlust.llm.types import LLMProvider

# OpenAI's reasoning_effort has no levels above "high".
_OPENAI_EFFORT = {
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
    Effort.XHIGH: "high",
    Effort.MAX: "high",
}


def default_model(provider: ProviderName) -> str:
    return {
        ProviderName.ANTHROPIC: anthropic_provider.DEFAULT_MODEL,
        ProviderName.OPENAI: openai_provider.DEFAULT_MODEL,
    }[provider]


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider chosen in settings.

    Raises:
        MissingAPIKeyError: if the selected provider has no API key.
    """
    api_key = settings.api_key()
    model = settings.llm_model or default_model(settings.llm_provider)

    if settings.llm_provider is ProviderName.ANTHROPIC:
        effort = (settings.llm_effort or Effort(anthropic_provider.DEFAULT_EFFORT)).value
        return anthropic_provider.AnthropicProvider(api_key, model=model, effort=effort)

    reasoning = _OPENAI_EFFORT[settings.llm_effort] if settings.llm_effort else None
    return openai_provider.OpenAIProvider(api_key, model=model, reasoning_effort=reasoning)
