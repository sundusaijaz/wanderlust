"""Command-line interface for Wanderlust."""

from __future__ import annotations

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from wanderlust import __version__
from wanderlust.config import Settings

app = typer.Typer(
    name="wanderlust",
    help="Plan trips with an AI agent grounded in Wikivoyage travel guides.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"wanderlust {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Wanderlust: an agentic trip planner."""


@app.command()
def config() -> None:
    """Show the active configuration (API keys are never printed)."""
    try:
        settings = Settings()
    except ValidationError as exc:
        console.print("[red]Invalid configuration:[/red]")
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            console.print(f"  • {field}: {error['msg']}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Wanderlust configuration", show_header=False)
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("LLM provider", settings.llm_provider.value)
    table.add_row("LLM model", settings.llm_model or "(provider default)")
    table.add_row(
        "API key",
        "[green]configured[/green]" if settings.has_api_key() else "[yellow]missing[/yellow]",
    )
    table.add_row("Wikivoyage language", settings.wikivoyage_language)
    table.add_row("Data directory", str(settings.data_dir))
    console.print(table)


@app.command()
def plan(
    request: Annotated[str, typer.Argument(help='What you want, e.g. "4 days in Lisbon".')],
) -> None:
    """Plan a trip from a natural-language request. (Coming in v0.1.)"""
    console.print(f"[bold]Trip request:[/bold] {request}")
    console.print(
        "[yellow]The planning agent is not built yet.[/yellow] Follow progress in ROADMAP.md."
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":  # pragma: no cover
    app()
