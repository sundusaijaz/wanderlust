"""Show the planning agent's progress and its finished itinerary in the terminal."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from wanderlust.agent.itinerary import DayPlan, TimeOfDay
from wanderlust.agent.planner import Plan, Progress
from wanderlust.agent.prompt import SUBMIT_TOOL
from wanderlust.llm.types import ToolCall, ToolResult

TIME_LABELS = {
    TimeOfDay.MORNING: "Morning",
    TimeOfDay.AFTERNOON: "Afternoon",
    TimeOfDay.EVENING: "Evening",
}
TOOL_ICONS = {
    "search_guide": "🔎",
    "geocode": "📍",
    "get_weather": "🌦 ",
    "get_distance": "📏",
    SUBMIT_TOOL: "📝",
}


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


def print_step(console: Console, call: ToolCall, result: ToolResult) -> None:
    icon = TOOL_ICONS.get(call.name, "🔧")
    outcome = summarize_result(call, result)
    style = "yellow" if result.is_error else "dim"
    marker = "✗" if result.is_error else "→"
    console.print(
        f"  {icon} {escape(describe_call(call))} [{style}]{marker} {escape(outcome)}[/{style}]",
        highlight=False,
    )


def print_itinerary(console: Console, plan: Plan) -> None:
    itinerary = plan.itinerary
    console.print(f"\n[bold magenta]{escape(itinerary.title)}[/bold magenta]")
    console.print(escape(itinerary.summary), highlight=False)
    for day in itinerary.days:
        _print_day(console, day)

    if itinerary.tips:
        console.print("\n[bold]Tips[/bold]")
        for tip in itinerary.tips:
            console.print(f"  • {escape(tip)}", highlight=False)
    if itinerary.assumptions:
        console.print("\n[bold]Assumptions[/bold]")
        for assumption in itinerary.assumptions:
            console.print(f"  [dim]• {escape(assumption)}[/dim]", highlight=False)

    console.print("\n[bold]Sources[/bold]")
    for source in plan.sources:
        console.print(f"[bold][{source.number}][/bold] {escape(source.label)}", highlight=False)
        console.print(f"    [dim]{escape(source.url)}[/dim]")
    console.print()
    console.print(f"[dim]{escape(_run_summary(plan))}[/dim]")
    console.print(
        "[dim]Wikivoyage content, CC BY-SA 4.0. Check prices and opening hours before you go.[/dim]"
    )
    if plan.used_weather:
        console.print("[dim]Weather data by Open-Meteo.com, CC BY 4.0.[/dim]")


def _print_day(console: Console, day: DayPlan) -> None:
    heading = f"Day {day.day}"
    if day.date:
        heading += f" · {day.date.strftime('%a %d %b %Y')}"
    heading += f" · {day.base}"
    console.print(f"\n[bold cyan]{escape(heading)}[/bold cyan]  [bold]{escape(day.title)}[/bold]")
    if day.weather:
        console.print(f"  [dim]Weather:[/dim] {escape(day.weather)}", highlight=False)

    table = Table(show_header=False, box=None, padding=(0, 1, 0, 2), expand=False)
    table.add_column("When", style="bold", no_wrap=True, min_width=9)
    table.add_column("What")
    for stop in day.stops:
        cites = "".join(f"[{n}]" for n in stop.sources)
        badge = "  [cyan]☂ indoor[/cyan]" if stop.indoor else ""
        table.add_row(
            TIME_LABELS[stop.time_of_day],
            f"[bold]{escape(stop.name)}[/bold] [dim]{escape(cites)}[/dim]{badge}\n"
            f"{escape(stop.description)}",
        )
    console.print(table)
    if day.getting_around:
        console.print(f"  [dim]Getting around:[/dim] {escape(day.getting_around)}", highlight=False)


def _run_summary(plan: Plan) -> str:
    return f"Planned by {plan.provider} · {plan.model}: {progress_summary(plan.progress)}."


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


def plan_json(plan: Plan) -> str:
    """The itinerary and its sources as JSON (for scripts and other tools)."""
    payload = {
        "request": plan.request,
        "itinerary": plan.itinerary.model_dump(mode="json"),
        "sources": [
            {"number": s.number, "label": s.label, "url": s.url, "license": "CC BY-SA 4.0"}
            for s in plan.sources
        ],
        "model": {"provider": plan.provider, "name": plan.model},
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _first_line(text: str, limit: int = 160) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."
