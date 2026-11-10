"""OpenAI models through the official OpenAI SDK (Chat Completions)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import openai

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

DEFAULT_MODEL = "gpt-5-mini"

_FINISH_REASONS = {
    "stop": StopReason.END,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


class OpenAIProvider:
    """Talks to the OpenAI Chat Completions API."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: Any = None,
    ) -> None:
        self._model = model
        self.reasoning_effort = reasoning_effort
        options: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if http_client is not None:
            options["http_client"] = http_client
        self._client = openai.OpenAI(**options)

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
            completion = self._client.chat.completions.create(**params)
        except openai.APIError as exc:
            raise _translate(exc) from exc
        return self._to_response(completion)

    def stream(self, conversation: Conversation, *, max_tokens: int = 16000) -> Iterator[str]:
        params = self._params(conversation, None, max_tokens)
        try:
            for chunk in self._client.chat.completions.create(**params, stream=True):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.APIError as exc:
            raise _translate(exc) from exc

    # -- request building -------------------------------------------------

    def _params(
        self, conversation: Conversation, tools: list[ToolSpec] | None, max_tokens: int
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages(conversation),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    @staticmethod
    def _messages(conversation: Conversation) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if conversation.system:
            messages.append({"role": "system", "content": conversation.system})
        for turn in conversation.turns:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantTurn):
                # Chat Completions keeps no hidden state, so the neutral fields are enough.
                response = turn.response
                message: dict[str, Any] = {"role": "assistant", "content": response.text or None}
                if response.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in response.tool_calls
                    ]
                messages.append(message)
            elif isinstance(turn, ToolResultsTurn):
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": f"Error: {result.content}"
                        if result.is_error
                        else result.content,
                    }
                    for result in turn.results
                )
        return messages

    # -- response parsing -------------------------------------------------

    def _to_response(self, completion: Any) -> LLMResponse:
        if not completion.choices:
            raise LLMError("OpenAI returned no choices")
        choice = completion.choices[0]
        message = choice.message

        calls = []
        for tool_call in message.tool_calls or []:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"OpenAI returned malformed arguments for {tool_call.function.name}"
                ) from exc
            calls.append(
                ToolCall(id=tool_call.id, name=tool_call.function.name, arguments=arguments)
            )

        stop = _FINISH_REASONS.get(choice.finish_reason or "", StopReason.OTHER)
        text = message.content or ""
        if getattr(message, "refusal", None):
            stop, text = StopReason.REFUSAL, message.refusal

        usage = completion.usage
        return LLMResponse(
            text=text,
            stop_reason=stop,
            provider=self.name,
            model=completion.model,
            tool_calls=calls,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            raw=message.to_dict(),
        )


def _translate(exc: openai.APIError) -> LLMError:
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError("The OpenAI API timed out. Try again.")
    if isinstance(exc, openai.APIConnectionError):
        return LLMConnectionError("Could not reach the OpenAI API. Check your connection.")
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return LLMAuthError("OpenAI rejected the API key. Check OPENAI_API_KEY.")
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError("OpenAI rate limit reached. Try again shortly.")
    if isinstance(exc, openai.APIStatusError):
        return LLMError(f"OpenAI API error (HTTP {exc.status_code}): {exc.message}")
    return LLMError(f"OpenAI API error: {exc}")
