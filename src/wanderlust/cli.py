"""Command-line interface for Wanderlust."""

from __future__ import annotations

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from wanderlust import __version__
from wanderlust.config import Settings
from wanderlust.wikivoyage.client import ArticleNotFoundError, WikivoyageError
from wanderlust.wikivoyage.models import Article, Section
from wanderlust.wikivoyage.service import build_loader

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


def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        console.print("[red]Invalid configuration:[/red]")
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            console.print(f"  • {field}: {error['msg']}")
        raise typer.Exit(code=1) from exc


@app.command()
def config() -> None:
    """Show the active configuration (API keys are never printed)."""
    settings = _load_settings()

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
    table.add_row("Guide cache lifetime", f"{settings.wikivoyage_cache_ttl_hours} hours")
    console.print(table)


@app.command()
def guide(
    place: Annotated[str, typer.Argument(help='Destination to look up, e.g. "Sintra".')],
    section: Annotated[
        str | None,
        typer.Option("--section", "-s", help='Show one section in full, e.g. "See" or "Eat".'),
    ] = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore the cache and download the guide again.")
    ] = False,
) -> None:
    """Show the Wikivoyage travel guide for a destination."""
    settings = _load_settings()
    loader = build_loader(settings)
    try:
        with console.status(f"Fetching the Wikivoyage guide for {escape(place)}..."):
            loaded = loader.load(place, refresh=refresh)
    except ArticleNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        if exc.suggestion:
            console.print(f'Try: [bold]wanderlust guide "{escape(exc.suggestion)}"[/bold]')
        raise typer.Exit(code=1) from exc
    except WikivoyageError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        loader.close()

    article = loaded.article
    if section is None:
        _print_overview(article)
        source_url = article.url
    else:
        found = article.section(section)
        if found is None:
            names = ", ".join(s.title for s in article.sections if s.level == 2)
            console.print(f"[red]No section called '{escape(section)}' in {article.title}.[/red]")
            console.print(f"Available sections: {escape(names)}")
            raise typer.Exit(code=1)
        _print_section(article, found)
        source_url = found.attribution.url

    origin = "from cache" if loaded.from_cache else "downloaded just now"
    console.print(
        f"[dim]Source: Wikivoyage, {escape(source_url)} · {article.license} · {origin}[/dim]"
    )


def _preview(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _print_overview(article: Article) -> None:
    if article.summary:
        console.print(
            Panel(escape(_preview(article.summary, 400)), title=f"[bold]{escape(article.title)}")
        )

    rows = [
        item
        for item in article.sections
        if not (item.is_empty and not article.subsections(item.title))
    ]
    labels = [item.title if item.level == 2 else f"  └ {item.title}" for item in rows]

    label_width = max(map(len, labels), default=7)
    # Borders and padding take 10 columns; give the preview whatever is left.
    preview_width = max(20, console.width - label_width - 6 - 10)

    table = Table(title="Sections", title_justify="left", show_lines=False)
    table.add_column("Section", style="bold", no_wrap=True, min_width=label_width)
    table.add_column("Places", justify="right", no_wrap=True, min_width=6)
    table.add_column("Preview", no_wrap=True, max_width=preview_width)
    for item, label in zip(rows, labels, strict=True):
        preview = item.text or ", ".join(listing.name for listing in item.listings)
        table.add_row(
            escape(label),
            str(len(item.listings)) if item.listings else "",
            escape(_preview(preview, preview_width)),
        )
    console.print(table)
    console.print(
        f"[dim]{len(article.listings)} places listed. "
        f'Show one section with: wanderlust guide "{escape(article.title)}" -s See[/dim]'
    )


def _print_section(article: Article, section: Section) -> None:
    parts = [section]
    if section.level == 2:
        parts += article.subsections(section.title)

    console.rule(f"[bold]{escape(article.title)} · {escape(section.title)}")
    for part in parts:
        if part is not section:
            console.print(f"\n[bold underline]{escape(part.title)}[/bold underline]")
        if part.text:
            console.print(escape(part.text))
        if part.listings:
            table = Table(show_lines=True, expand=True)
            table.add_column("Place", style="bold", ratio=2)
            table.add_column("Details", ratio=2)
            table.add_column("About", ratio=5)
            for listing in part.listings:
                name = escape(listing.name)
                if listing.alt:
                    name += f"\n[dim]{escape(listing.alt)}[/dim]"
                details = [
                    f"{label}: {escape(value)}"
                    for label, value in (
                        ("Price", listing.price),
                        ("Hours", listing.hours),
                        ("Address", listing.address),
                    )
                    if value
                ]
                if listing.has_coordinates:
                    details.append("[green]📍 on map[/green]")
                table.add_row(name, "\n".join(details), escape(_preview(listing.description, 220)))
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
