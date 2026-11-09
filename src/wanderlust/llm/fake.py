"""A scripted provider for tests and offline demos."""

from __future__ import annotations

import copy
import re
from collections import deque
from collections.abc import Iterable, Iterator

from wanderlust.llm.types import (
    Conversation,
    LLMError,
    LLMResponse,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
    UserTurn,
)


class ScriptedProvider:
    """Replies with pre-written responses, in order, and records every request.

    Each scripted item is either plain text (an ordinary answer) or a full
    :class:`LLMResponse` (for example one that asks for tool calls).
    """

    name = "fake"

    def __init__(self, script: Iterable[str | LLMResponse], *, model: str = "scripted") -> None:
        self._model = model
        self._script: deque[str | LLMResponse] = deque(script)
        self.requests: list[tuple[Conversation, list[ToolSpec]]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def remaining(self) -> int:
        return len(self._script)

    def complete(
        self,
        conversation: Conversation,
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        self.requests.append((copy.deepcopy(conversation), list(tools or [])))
        if not self._script:
            raise LLMError("ScriptedProvider has no responses left")
        item = self._script.popleft()
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(
            text=item,
            stop_reason=StopReason.END,
            provider=self.name,
            model=self._model,
            usage=Usage(input_tokens=_count_words(conversation), output_tokens=len(item.split())),
        )

    def stream(self, conversation: Conversation, *, max_tokens: int = 16000) -> Iterator[str]:
        response = self.complete(conversation, max_tokens=max_tokens)
        # Split into words but keep the whitespace, so joining the pieces restores the text.
        yield from re.findall(r"\S+\s*|\s+", response.text)

    @staticmethod
    def tool_request(*calls: ToolCall, text: str = "") -> LLMResponse:
        """Convenience for scripting a reply that asks for tool calls."""
        return LLMResponse(
            text=text,
            stop_reason=StopReason.TOOL_USE,
            provider=ScriptedProvider.name,
            model="scripted",
            tool_calls=list(calls),
        )


def _count_words(conversation: Conversation) -> int:
    return sum(len(turn.text.split()) for turn in conversation.turns if isinstance(turn, UserTurn))
