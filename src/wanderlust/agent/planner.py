"""The planning agent: research with tools, then submit a cited itinerary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from pydantic import ValidationError

from wanderlust.agent.guides import GuideLibrary, SourceBook, guide_tools
from wanderlust.agent.itinerary import Itinerary
from wanderlust.agent.prompt import NUDGE, SUBMIT_TOOL, request_message, system_prompt
from wanderlust.llm.types import (
    Conversation,
    LLMProvider,
    LLMResponse,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from wanderlust.qa.sources import Source
from wanderlust.retrieval.search import SearchMode
from wanderlust.tools.registry import ToolRegistry, json_schema, validation_problems

DEFAULT_MAX_STEPS = 16
DEFAULT_MAX_TOKENS = 16000

SUBMIT_SPEC = ToolSpec(
    name=SUBMIT_TOOL,
    description=(
        "Submit the finished itinerary. Every stop must cite passages returned by "
        "search_guide. If the result lists problems, fix them and submit again."
    ),
    parameters=json_schema(Itinerary),
)


# -- what the agent reports while it works -------------------------------------


@dataclass(frozen=True)
class ModelReplied:
    step: int
    response: LLMResponse


@dataclass(frozen=True)
class ToolStarted:
    call: ToolCall


@dataclass(frozen=True)
class ToolFinished:
    call: ToolCall
    result: ToolResult


AgentEvent = ModelReplied | ToolStarted | ToolFinished


@dataclass(frozen=True)
class ToolStep:
    call: ToolCall
    result: ToolResult


@dataclass
class Progress:
    """Everything the agent did, kept even when planning fails."""

    tool_steps: list[ToolStep] = field(default_factory=list)
    model_calls: int = 0
    usage: Usage = field(default_factory=Usage)

    def count(self, response: LLMResponse) -> None:
        self.model_calls += 1
        self.usage = Usage(
            input_tokens=self.usage.input_tokens + response.usage.input_tokens,
            output_tokens=self.usage.output_tokens + response.usage.output_tokens,
        )

    def calls_to(self, name: str) -> list[ToolStep]:
        return [step for step in self.tool_steps if step.call.name == name]


@dataclass(frozen=True)
class Plan:
    request: str
    itinerary: Itinerary
    sources: list[Source]
    """The passages the itinerary cites, in order of first citation."""
    progress: Progress
    provider: str
    model: str

    def source(self, number: int) -> Source | None:
        return next((s for s in self.sources if s.number == number), None)

    @property
    def used_weather(self) -> bool:
        return any(not step.result.is_error for step in self.progress.calls_to("get_weather"))


class PlanningError(RuntimeError):
    """The agent could not produce an itinerary."""

    def __init__(self, message: str, progress: Progress, *, model_text: str = "") -> None:
        super().__init__(message)
        self.progress = progress
        self.model_text = model_text


# -- the agent ------------------------------------------------------------------


class PlanningAgent:
    """Runs the tool-calling loop until the model submits a valid itinerary.

    Each model reply may ask for several tools; their results go back in one
    turn. Tool problems become error results the model can react to, and a
    rejected itinerary comes back with its problems listed. The loop stops
    after ``max_steps`` model calls.
    """

    def __init__(
        self,
        provider: LLMProvider,
        library: GuideLibrary,
        travel_tools: ToolRegistry | None = None,
        *,
        mode: SearchMode | str = SearchMode.HYBRID,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        today: Callable[[], date] = date.today,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> None:
        if max_steps < 2:
            raise ValueError("max_steps must be at least 2 (research, then submit)")
        self.provider = provider
        self.library = library
        self.travel_tools = travel_tools or ToolRegistry([])
        self.mode = SearchMode(mode)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.today = today
        self.on_event = on_event or (lambda event: None)

    def plan(self, request: str) -> Plan:
        if not request.strip():
            raise ValueError("the trip request must not be empty")

        book = SourceBook()
        tools = guide_tools(self.library, book, mode=self.mode) + self.travel_tools
        if SUBMIT_TOOL in tools:
            raise ValueError(f"'{SUBMIT_TOOL}' is reserved for the agent")
        specs = [*tools.specs, SUBMIT_SPEC]

        conversation = Conversation(system=system_prompt(self.today()))
        conversation.add_user(request_message(request))
        progress = Progress()
        nudged = False

        for step in range(1, self.max_steps + 1):
            response = self.provider.complete(conversation, tools=specs, max_tokens=self.max_tokens)
            progress.count(response)
            conversation.add_response(response)
            self.on_event(ModelReplied(step, response))

            if response.stop_reason is StopReason.REFUSAL:
                raise PlanningError(
                    "The model declined to plan this trip.", progress, model_text=response.text
                )
            if not response.tool_calls:
                if response.stop_reason is StopReason.MAX_TOKENS:
                    raise PlanningError(
                        "The model ran out of room before finishing the plan; "
                        "try a shorter trip or raise the token limit.",
                        progress,
                        model_text=response.text,
                    )
                if nudged:
                    raise PlanningError(
                        "The model stopped without submitting an itinerary.",
                        progress,
                        model_text=response.text,
                    )
                nudged = True
                conversation.add_user(NUDGE)
                continue

            results: list[ToolResult] = []
            accepted: Itinerary | None = None
            for call in response.tool_calls:
                self.on_event(ToolStarted(call))
                if call.name == SUBMIT_TOOL:
                    result, itinerary = check_submission(call, book)
                    accepted = itinerary or accepted
                else:
                    result = tools.run(call)
                progress.tool_steps.append(ToolStep(call, result))
                results.append(result)
                self.on_event(ToolFinished(call, result))
            conversation.add_tool_results(results)

            if accepted is not None:
                return Plan(
                    request=request,
                    itinerary=accepted,
                    sources=book.select(accepted.cited_numbers()),
                    progress=progress,
                    provider=self.provider.name,
                    model=self.provider.model,
                )

        raise PlanningError(
            f"No itinerary after {self.max_steps} steps. Try a simpler request "
            "or allow more steps with --max-steps.",
            progress,
        )


def check_submission(call: ToolCall, book: SourceBook) -> tuple[ToolResult, Itinerary | None]:
    """Validate a submitted itinerary, including that its citations are real."""

    def reject(message: str) -> tuple[ToolResult, None]:
        return ToolResult(call_id=call.id, content=message, is_error=True), None

    try:
        itinerary = Itinerary.model_validate(call.arguments)
    except ValidationError as exc:
        return reject(f"The itinerary has problems: {validation_problems(exc)}")

    if len(book) == 0:
        return reject(
            "No guide passages have been retrieved yet. Search the guides with "
            "search_guide first; every stop must cite the passages that mention it."
        )
    unknown = [n for n in itinerary.cited_numbers() if n not in book]
    if unknown:
        numbers = ", ".join(f"[{n}]" for n in unknown)
        return reject(
            f"The itinerary cites {numbers}, but search_guide never returned "
            f"{'that passage' if len(unknown) == 1 else 'those passages'} "
            f"(known passages: 1 to {len(book)}). Cite only retrieved passages, "
            "or search for those places first."
        )
    return ToolResult(call_id=call.id, content="Itinerary accepted."), itinerary
