"""Provider-neutral types for talking to language models.

Wanderlust code builds a :class:`Conversation` and gets back an
:class:`LLMResponse`; each provider translates to and from its own API.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class StopReason(StrEnum):
    END = "end"  # the model finished its answer
    TOOL_USE = "tool_use"  # the model wants tools to be run
    MAX_TOKENS = "max_tokens"  # the answer was cut off
    REFUSAL = "refusal"  # the model (or a safety system) declined
    OTHER = "other"


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call, described with a JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    stop_reason: StopReason
    provider: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = field(default=None, repr=False, compare=False)
    """The provider's own assistant message, replayed unchanged on the next turn
    (it may carry blocks such as thinking that must be passed back as-is)."""


@dataclass(frozen=True)
class UserTurn:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    response: LLMResponse


@dataclass(frozen=True)
class ToolResultsTurn:
    results: list[ToolResult]


Turn = UserTurn | AssistantTurn | ToolResultsTurn


@dataclass
class Conversation:
    """An append-only transcript that any provider can send."""

    system: str = ""
    turns: list[Turn] = field(default_factory=list)

    def add_user(self, text: str) -> Conversation:
        self.turns.append(UserTurn(text))
        return self

    def add_response(self, response: LLMResponse) -> Conversation:
        self.turns.append(AssistantTurn(response))
        return self

    def add_tool_results(self, results: list[ToolResult]) -> Conversation:
        if not results:
            raise ValueError("at least one tool result is required")
        self.turns.append(ToolResultsTurn(list(results)))
        return self


class LLMError(RuntimeError):
    """A language model request failed."""


class LLMAuthError(LLMError):
    """The API key is missing, invalid or lacks permission."""


class LLMRateLimitError(LLMError):
    """The provider is rate limiting or overloaded; try again later."""


class LLMTimeoutError(LLMError):
    """The request took too long."""


class LLMConnectionError(LLMError):
    """The provider could not be reached."""


@runtime_checkable
class LLMProvider(Protocol):
    """What Wanderlust needs from a language model provider."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(
        self,
        conversation: Conversation,
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        """Send the conversation and return the full reply (text and/or tool calls)."""
        ...

    def stream(self, conversation: Conversation, *, max_tokens: int = 16000) -> Iterator[str]:
        """Send the conversation and yield the reply's text as it is generated."""
        ...
