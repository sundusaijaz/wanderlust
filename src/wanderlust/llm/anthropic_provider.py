"""Claude models through the official Anthropic SDK."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anthropic

from wanderlust.llm.types import (
    AssistantTurn,
    Conversation,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    StopReason,
    ToolCall,
    ToolResultsTurn,
    ToolSpec,
    Usage,
    UserTurn,
)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"

# Server-side refusal fallback: if a safety classifier declines, the API re-runs
# the request on a suitable fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)

_STOP_REASONS = {
    "end_turn": StopReason.END,
    "stop_sequence": StopReason.END,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}

# Blocks that must not be echoed if they come before a mid-answer fallback.
_PRE_FALLBACK_DROP = frozenset({"thinking", "redacted_thinking", "tool_use", "server_tool_use"})


class AnthropicProvider:
    """Talks to the Claude Messages API.

    Adaptive thinking is always on for current models; ``effort`` controls how
    much the model thinks. Assistant turns are replayed exactly as returned,
    so thinking blocks travel back unchanged.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        fallbacks: bool = True,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: Any = None,
    ) -> None:
        self._model = model
        self.effort = effort
        self.fallbacks = fallbacks and model in FALLBACK_MODELS
        options: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if http_client is not None:
            options["http_client"] = http_client
        self._client = anthropic.Anthropic(**options)

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        conversation: Conversation,
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        params = self._params(conversation, tools, max_tokens)
        try:
            message = self._client.beta.messages.create(**params)
        except anthropic.APIError as exc:
            raise _translate(exc) from exc
        return self._to_response(message)

    def stream(self, conversation: Conversation, *, max_tokens: int = 16000) -> Iterator[str]:
        params = self._params(conversation, None, max_tokens)
        try:
            with self._client.beta.messages.stream(**params) as stream:
                yield from stream.text_stream
        except anthropic.APIError as exc:
            raise _translate(exc) from exc

    # -- request building -------------------------------------------------

    def _params(
        self, conversation: Conversation, tools: list[ToolSpec] | None, max_tokens: int
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": self._messages(conversation),
            "output_config": {"effort": self.effort},
        }
        if conversation.system:
            params["system"] = conversation.system
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if self.fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _messages(self, conversation: Conversation) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in conversation.turns:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantTurn):
                messages.append({"role": "assistant", "content": self._assistant_content(turn)})
            elif isinstance(turn, ToolResultsTurn):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result.call_id,
                                "content": result.content,
                                "is_error": result.is_error,
                            }
                            for result in turn.results
                        ],
                    }
                )
        return messages

    def _assistant_content(self, turn: AssistantTurn) -> list[dict[str, Any]]:
        response = turn.response
        if response.provider == self.name and isinstance(response.raw, list):
            return echo_blocks(response.raw)
        # A turn produced by another provider: rebuild it from the neutral fields.
        blocks: list[dict[str, Any]] = []
        if response.text:
            blocks.append({"type": "text", "text": response.text})
        blocks.extend(
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in response.tool_calls
        )
        return blocks

    # -- response parsing -------------------------------------------------

    def _to_response(self, message: Any) -> LLMResponse:
        text = "".join(block.text for block in message.content if block.type == "text")
        calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
            for block in message.content
            if block.type == "tool_use"
        ]
        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            stop_reason=_STOP_REASONS.get(message.stop_reason or "", StopReason.OTHER),
            provider=self.name,
            model=message.model,
            tool_calls=calls,
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ),
            raw=[block.to_dict() for block in message.content],
        )


def echo_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Content blocks to send back for an earlier assistant turn.

    Blocks are passed back unchanged, except around a mid-answer refusal
    fallback: thinking and tool-use blocks before the last ``fallback`` marker
    are dropped (they belong to the declined attempt), and the marker itself is
    left out.
    """
    boundary = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    echoed = []
    for i, block in enumerate(blocks):
        kind = block.get("type")
        if kind == "fallback" or (i < boundary and kind in _PRE_FALLBACK_DROP):
            continue
        echoed.append(block)
    return echoed


def _translate(exc: anthropic.APIError) -> LLMError:
    """Map SDK exceptions onto Wanderlust's provider-neutral errors."""
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError("The Anthropic API timed out. Try again.")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMConnectionError("Could not reach the Anthropic API. Check your connection.")
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return LLMAuthError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.")
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code in (429, 529):
        return LLMRateLimitError("Anthropic is rate limiting or overloaded. Try again shortly.")
    if isinstance(exc, anthropic.APIStatusError):
        return LLMError(f"Anthropic API error (HTTP {exc.status_code}): {exc.message}")
    return LLMError(f"Anthropic API error: {exc}")
