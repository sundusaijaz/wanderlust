"""Show the planning agent's progress and its finished itinerary in the terminal."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from wanderlust.agent.describe import (
    describe_call,
    plan_json,
    progress_summary,
    summarize_result,
)
from wanderlust.agent.itinerary import DayPlan, TimeOfDay
from wanderlust.agent.planner import Plan
from wanderlust.agent.prompt import SUBMIT_TOOL
from wanderlust.llm.types import ToolCall, ToolResult

__all__ = [
    "describe_call",
    "plan_json",
    "print_itinerary",
    "print_step",
    "progress_summary",
    "summarize_result",
]

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
    console.print(
        f"\n[bold cyan]{escape(heading)}[/bold cyan]  [bold]{escape(day.title)}[/bold]",
        highlight=False,
    )
    if day.weather:
        console.print(f"  [dim]Weather:[/dim] {escape(day.weather)}", highlight=False)

    table = Table(show_header=False, box=None, padding=(0, 1, 0, 2), expand=False)
    table.add_column("When", style="bold", no_wrap=True, min_width=9)
    table.add_column("What")
    for stop in day.stops:
        cites = "".join(f"[{n}]" for n in stop.sources)
        badge = "  [cyan]☔ indoor[/cyan]" if stop.indoor else ""
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
