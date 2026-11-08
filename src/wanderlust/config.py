"""Application settings, loaded from environment variables and an optional .env file."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Language model providers Wanderlust can talk to."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider has no API key configured."""


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be set with an environment variable prefixed with
    ``WANDERLUST_`` (for example ``WANDERLUST_LLM_PROVIDER=anthropic``).
    Provider API keys use their standard names (``OPENAI_API_KEY``,
    ``ANTHROPIC_API_KEY``) so existing setups work unchanged.
    """

    model_config = SettingsConfigDict(
        env_prefix="WANDERLUST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str | None = Field(
        default=None,
        description="Model name. Leave empty to use the provider's default.",
    )

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )

    wikivoyage_language: str = Field(
        default="en",
        pattern=r"^[a-z]{2,3}$",
        description="Language edition of Wikivoyage to read, e.g. 'en' or 'de'.",
    )
    data_dir: Path = Field(
        default=Path(".wanderlust"),
        description="Where cached guides and search indexes are stored.",
    )

    def api_key(self) -> str:
        """Return the API key for the selected provider.

        Raises:
            MissingAPIKeyError: if no key is configured for that provider.
        """
        key = {
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
        }[self.llm_provider]
        if key is None or not key.get_secret_value().strip():
            env_name = f"{self.llm_provider.value.upper()}_API_KEY"
            raise MissingAPIKeyError(
                f"No API key found for provider '{self.llm_provider.value}'. "
                f"Set {env_name} in your environment or .env file."
            )
        return key.get_secret_value()

    def has_api_key(self) -> bool:
        """Whether the selected provider has an API key configured."""
        try:
            self.api_key()
        except MissingAPIKeyError:
            return False
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them once."""
    return Settings()
