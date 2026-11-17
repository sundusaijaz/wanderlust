"""Plain-text descriptions of what the planning agent does, for any interface."""

from __future__ import annotations

import json
from typing import Any

from wanderlust.agent.planner import Plan, Progress
from wanderlust.agent.prompt import SUBMIT_TOOL
from wanderlust.llm.types import ToolCall, ToolResult


def describe_call(call: ToolCall) -> str:
    """What a tool call is doing, in plain words (for the progress spinner)."""
    args = call.arguments
    match call.name:
        case "search_guide":
            return f"Searching the {args.get('destination', '?')} guide for '{args.get('query')}'"
        case "geocode":
            return f"Looking up {args.get('place', '?')}"
        case "get_weather":
            return f"Checking the weather from {args.get('start_date', '?')}"
        case "get_distance":
            return "Estimating travel times"
        case name if name == SUBMIT_TOOL:
            return "Checking the itinerary"
        case _:
            return f"Running {call.name}"


def summarize_result(call: ToolCall, result: ToolResult) -> str:
    """A one-line outcome of a tool call (for --verbose)."""
    if result.is_error:
        return _first_line(result.content)
    if call.name == "search_guide":
        found = result.content.count("<passage ")
        return f"{found} passage{'s' if found != 1 else ''}" if found else "nothing found"
    if call.name == SUBMIT_TOOL:
        return "accepted"
    data = _json(result.content)
    if call.name == "geocode" and data:
        return data["matches"][0]["label"]
    if call.name == "get_weather" and data:
        wet = sum(1 for day in data["days"] if day.get("plan_indoor_alternatives"))
        kind = "forecast" if data["kind"] == "forecast" else "typical weather"
        return f"{kind}, {len(data['days'])} days, {wet} wet"
    if call.name == "get_distance" and data:
        return f"{data['straight_line_km']:.1f} km"
    return _first_line(result.content)


def progress_summary(progress: Progress) -> str:
    calls = len(progress.tool_steps)
    summary = (
        f"{progress.model_calls} model call{'s' if progress.model_calls != 1 else ''}, "
        f"{calls} tool call{'s' if calls != 1 else ''}"
    )
    tokens = progress.usage.input_tokens + progress.usage.output_tokens
    if tokens:
        summary += f", {tokens:,} tokens"
    return summary


def plan_payload(plan: Plan) -> dict[str, Any]:
    """The itinerary, its sources and the run as plain data."""
    return {
        "request": plan.request,
        "itinerary": plan.itinerary.model_dump(mode="json"),
        "sources": [
            {"number": s.number, "label": s.label, "url": s.url, "license": "CC BY-SA 4.0"}
            for s in plan.sources
        ],
        "model": {"provider": plan.provider, "name": plan.model},
    }


def plan_json(plan: Plan) -> str:
    """The itinerary and its sources as JSON (for scripts and other tools)."""
    return json.dumps(plan_payload(plan), indent=2, ensure_ascii=False)


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _first_line(text: str, limit: int = 160) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."
