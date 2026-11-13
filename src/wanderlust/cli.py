"""Command-line interface for Wanderlust."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from wanderlust import __version__
from wanderlust.agent.guides import GuideLibrary, index_opener
from wanderlust.agent.planner import (
    DEFAULT_MAX_STEPS,
    AgentEvent,
    ModelReplied,
    PlanningAgent,
    PlanningError,
    ToolStarted,
)
from wanderlust.config import MissingAPIKeyError, Settings
from wanderlust.llm.factory import build_provider, default_model
from wanderlust.llm.types import Conversation, LLMError
from wanderlust.plan_view import (
    describe_call,
    plan_json,
    print_itinerary,
    print_step,
    progress_summary,
)
from wanderlust.qa.answerer import DEFAULT_PASSAGES, Answer, GroundedAnswerer
from wanderlust.retrieval.builder import BuildReport, build_index
from wanderlust.retrieval.embeddings import Embedder, EmbeddingError, build_embedder
from wanderlust.retrieval.index import GuideIndex, IndexStore
from wanderlust.retrieval.search import Searcher, SearchError, SearchMode
from wanderlust.streaming import WordWrapper
from wanderlust.tools.distance import distance_report
from wanderlust.tools.geocoding import Geocoder, Place
from wanderlust.tools.http import ToolError
from wanderlust.tools.registry import build_tools_client, travel_tools
from wanderlust.tools.weather import WeatherService
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
    table.add_row(
        "LLM model", settings.llm_model or f"{default_model(settings.llm_provider)} (default)"
    )
    table.add_row(
        "LLM effort", settings.llm_effort.value if settings.llm_effort else "(provider default)"
    )
    table.add_row(
        "API key",
        "[green]configured[/green]" if settings.has_api_key() else "[yellow]missing[/yellow]",
    )
    table.add_row("Wikivoyage language", settings.wikivoyage_language)
    table.add_row("Data directory", str(settings.data_dir))
    table.add_row("Guide cache lifetime", f"{settings.wikivoyage_cache_ttl_hours} hours")
    try:
        embedder = build_embedder(settings)
        semantic = embedder.name if embedder else "[yellow]off (keyword search only)[/yellow]"
    except MissingAPIKeyError:
        semantic = "[red]openai selected but OPENAI_API_KEY is missing[/red]"
    table.add_row("Semantic search", f"{settings.embedding_provider.value} → {semantic}")
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


def _embedder_or_exit(settings: Settings) -> Embedder | None:
    try:
        return build_embedder(settings)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc


def _build_and_save(
    place: str,
    settings: Settings,
    embedder: Embedder | None,
    *,
    refresh: bool = False,
    follow_districts: bool = True,
) -> BuildReport:
    loader = build_loader(settings)
    try:
        with console.status("Building the search index...") as status:
            report = build_index(
                place,
                loader,
                embedder,
                refresh=refresh,
                follow_districts=follow_districts,
                on_progress=lambda message: status.update(escape(message) + "..."),
            )
    except ArticleNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        if exc.suggestion:
            console.print(f'Try: [bold]wanderlust index "{escape(exc.suggestion)}"[/bold]')
        raise typer.Exit(code=1) from exc
    except (WikivoyageError, EmbeddingError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        loader.close()
    IndexStore(settings.data_dir).save(report.index)
    return report


@app.command()
def index(
    place: Annotated[str, typer.Argument(help='Destination to index, e.g. "Lisbon".')],
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Download the guides again before indexing.")
    ] = False,
    districts: Annotated[
        bool,
        typer.Option("--districts/--no-districts", help="Also index district articles."),
    ] = True,
) -> None:
    """Build the search index for a destination (and its districts)."""
    settings = _load_settings()
    embedder = _embedder_or_exit(settings)
    report = _build_and_save(place, settings, embedder, refresh=refresh, follow_districts=districts)

    meta = report.index.meta
    console.print(
        f"[green]✓[/green] Indexed [bold]{escape(meta.destination)}[/bold]: "
        f"{meta.chunk_count} passages from {len(report.articles)} "
        f"article{'s' if len(report.articles) != 1 else ''}."
    )
    if len(report.articles) > 1:
        names = ", ".join(a.title.split("/", 1)[-1] for a in report.articles[1:])
        console.print(f"  Districts: {escape(names)}")
    if report.skipped:
        console.print(f"[dim]  Skipped: {escape(', '.join(report.skipped))}[/dim]")
    if meta.embedder:
        console.print(f"  Semantic search: {escape(meta.embedder)} ({meta.dims} dimensions)")
    else:
        console.print(
            "  [yellow]Keyword search only.[/yellow] Set OPENAI_API_KEY or install "
            '"wanderlust\\[local]" for semantic search.'
        )
    console.print(f'[dim]Try: wanderlust search "{escape(meta.destination)}" "museums"[/dim]')


def _index_for(place: str, settings: Settings, embedder: Embedder | None) -> GuideIndex:
    """The saved index for a destination, building and saving it first if needed."""
    guide_index = IndexStore(settings.data_dir).load(settings.wikivoyage_language, place)
    if guide_index is None:
        console.print(f"[dim]No index for {escape(place)} yet; building it now.[/dim]")
        guide_index = _build_and_save(place, settings, embedder).index
    return guide_index


@app.command()
def search(
    place: Annotated[str, typer.Argument(help='Destination, e.g. "Lisbon".')],
    query: Annotated[str, typer.Argument(help='What to look for, e.g. "rainy day museums".')],
    top: Annotated[int, typer.Option("--top", "-k", min=1, max=50, help="Number of results.")] = 5,
    mode: Annotated[
        SearchMode, typer.Option("--mode", "-m", help="hybrid, keyword or vector.")
    ] = SearchMode.HYBRID,
) -> None:
    """Search a destination's guides. Builds the index first if needed."""
    settings = _load_settings()
    embedder = _embedder_or_exit(settings)
    guide_index = _index_for(place, settings, embedder)

    try:
        response = Searcher(guide_index, embedder).search(query, k=top, mode=mode)
    except SearchError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    if response.note:
        console.print(f"[yellow]{escape(response.note)}[/yellow]")
    if not response.results:
        console.print(
            f"No passages about '{escape(query)}' in the {guide_index.meta.destination} guide."
        )
        raise typer.Exit(code=1)
    _print_results(guide_index, response.results, response.mode)


def _print_results(guide_index: GuideIndex, results, mode: SearchMode) -> None:
    for n, result in enumerate(results, start=1):
        chunk = result.chunk
        title = escape(chunk.breadcrumb)
        if chunk.listing_name:
            title += f" · [cyan]{escape(chunk.listing_name)}[/cyan]"
        ranks = [
            f"{label} #{rank}"
            for label, rank in (("keyword", result.keyword_rank), ("vector", result.vector_rank))
            if rank is not None
        ]
        console.print(f"\n[bold]{n}. {title}[/bold]  [dim]{' · '.join(ranks)}[/dim]")
        console.print(escape(_preview(chunk.text, 320)), highlight=False)
        console.print(f"[dim]{escape(chunk.attribution.url)}[/dim]")
    console.print(
        f"\n[dim]{mode.value} search over {guide_index.meta.chunk_count} passages · "
        f"Wikivoyage content, {chunk.attribution.license}[/dim]"
    )


@app.command()
def ask(
    place: Annotated[str, typer.Argument(help='Destination, e.g. "Lisbon".')],
    question: Annotated[
        str, typer.Argument(help='Your question, e.g. "Where can I try custard tarts?"')
    ],
    passages: Annotated[
        int,
        typer.Option(
            "--passages", "-k", min=1, max=20, help="How many guide passages to give the model."
        ),
    ] = DEFAULT_PASSAGES,
    mode: Annotated[
        SearchMode, typer.Option("--mode", "-m", help="hybrid, keyword or vector.")
    ] = SearchMode.HYBRID,
    show_passages: Annotated[
        bool, typer.Option("--show-passages", help="Also print the passages sent to the model.")
    ] = False,
) -> None:
    """Answer a question from a destination's travel guide, citing its sources."""
    settings = _load_settings()
    embedder = _embedder_or_exit(settings)
    try:
        provider = build_provider(settings)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    guide_index = _index_for(place, settings, embedder)
    destination = guide_index.meta.destination
    answerer = GroundedAnswerer(
        Searcher(guide_index, embedder), provider, passages=passages, mode=mode
    )
    try:
        prepared = answerer.prepare(question)
    except (SearchError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    if prepared.search.note:
        console.print(f"[yellow]{escape(prepared.search.note)}[/yellow]")
    if not prepared.has_context:
        console.print(
            f"No passages in the {escape(destination)} guide match that question, "
            "so there is nothing to answer from. Try other words or 'wanderlust guide'."
        )
        raise typer.Exit(code=1)

    if show_passages:
        for source in prepared.sources:
            console.print(f"[bold][{source.number}][/bold] {escape(source.label)}")
            console.print(f"[dim]{escape(_preview(source.chunk.text, 240))}[/dim]")
        console.print()

    console.print(
        f"[dim]{escape(provider.name)} · {escape(provider.model)} · "
        f"{len(prepared.sources)} passages from the {escape(destination)} guide[/dim]\n"
    )
    pieces: list[str] = []
    wrapper = _wrapper()
    try:
        for piece in answerer.stream(prepared):
            pieces.append(piece)
            wrapper.feed(piece)
        wrapper.close()
    except LLMError as exc:
        console.print(f"\n[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    console.print()
    _print_sources(answerer.finish(prepared, "".join(pieces)))


def _wrapper() -> WordWrapper:
    """Write streamed model text to the console, wrapped between words."""

    def write(text: str) -> None:
        console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    return WordWrapper(write, width=max(20, console.width))


def _print_sources(answer: Answer) -> None:
    report = answer.citations
    if report.cited:
        console.print("\n[bold]Sources[/bold]")
        for source in report.cited:
            console.print(f"[bold][{source.number}][/bold] {escape(source.label)}")
            console.print(f"    [dim]{escape(source.url)}[/dim]")
    elif answer.covered:
        console.print("\n[yellow]The answer cites no passages. It was given these:[/yellow]")
        for source in answer.sources:
            console.print(f"[{source.number}] {escape(source.label)}", markup=False)
    if report.unknown:
        numbers = ", ".join(f"[{n}]" for n in report.unknown)
        console.print(
            f"[yellow]Note: the answer cites {numbers}, which is not one of the passages "
            "it was given. Treat that part with care.[/yellow]"
        )
    console.print(
        "\n[dim]Wikivoyage content, CC BY-SA 4.0. "
        "Check prices and opening hours before you go.[/dim]"
    )


def _locate(geocoder: Geocoder, name: str) -> Place:
    try:
        return geocoder.locate(name)
    except (ToolError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def weather(
    place: Annotated[str, typer.Argument(help='Where, e.g. "Lisbon".')],
    start: Annotated[
        str | None, typer.Option("--from", help="First day (YYYY-MM-DD). Default: today.")
    ] = None,
    days: Annotated[int, typer.Option("--days", "-d", min=1, max=16, help="Number of days.")] = 5,
) -> None:
    """Weather for a place: a forecast for the next 15 days, typical weather after that."""
    settings = _load_settings()
    client = build_tools_client(settings.data_dir)
    try:
        service = WeatherService(client)
        try:
            first = date.fromisoformat(start) if start else service.today()
        except ValueError as exc:
            console.print(f"[red]Invalid date '{escape(start or '')}'; use YYYY-MM-DD.[/red]")
            raise typer.Exit(code=1) from exc
        location = _locate(Geocoder(client, language=settings.wikivoyage_language), place)
        try:
            report = service.for_dates(
                location.latitude, location.longitude, first, first + timedelta(days=days - 1)
            )
        except (ToolError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(code=1) from exc
    finally:
        client.close()

    title = "Forecast" if report.kind == "forecast" else "Typical weather"
    table = Table(title=f"{title} for {escape(location.label)}", title_justify="left")
    table.add_column("Date", no_wrap=True)
    table.add_column("Weather")
    table.add_column("Temp °C", justify="right", no_wrap=True)
    table.add_column("Rain", justify="right", no_wrap=True)
    table.add_column("Chance", justify="right", no_wrap=True)
    table.add_column("")
    for day in report.days:
        temps = (
            f"{day.temp_min_c:.0f} to {day.temp_max_c:.0f}"
            if day.temp_min_c is not None and day.temp_max_c is not None
            else "?"
        )
        rain = f"{day.precipitation_mm:.1f} mm" if day.precipitation_mm is not None else "?"
        chance = (
            f"{day.precipitation_probability}%"
            if day.precipitation_probability is not None
            else "?"
        )
        table.add_row(
            day.date.strftime("%a %d %b"),
            escape(day.description),
            temps,
            rain,
            chance,
            "[cyan]☔ indoor day[/cyan]" if day.is_wet else "",
        )
    console.print(table)
    if report.note:
        console.print(f"[yellow]{escape(report.note)}[/yellow]")
    console.print(f"[dim]{escape(report.attribution)}[/dim]")


@app.command()
def distance(
    origin: Annotated[str, typer.Argument(help='Start, e.g. "Lisbon".')],
    destination: Annotated[str, typer.Argument(help='End, e.g. "Sintra".')],
) -> None:
    """Rough distance and travel times between two places."""
    settings = _load_settings()
    client = build_tools_client(settings.data_dir)
    try:
        geocoder = Geocoder(client, language=settings.wikivoyage_language)
        start, end = _locate(geocoder, origin), _locate(geocoder, destination)
    finally:
        client.close()

    report = distance_report(start.latitude, start.longitude, end.latitude, end.longitude)
    console.print(
        f"[bold]{escape(start.label)}[/bold] → [bold]{escape(end.label)}[/bold]: "
        f"{report.straight_line_km:.1f} km in a straight line"
    )
    labels = {"walk": "🚶 Walking", "transit": "🚆 Public transport", "drive": "🚗 Driving"}
    for estimate in report.estimates:
        hours, minutes = divmod(estimate.minutes, 60)
        duration = f"{hours} h {minutes} min" if hours else f"{minutes} min"
        console.print(f"  {labels[estimate.mode.value]}: about {duration}")
    console.print(f"[dim]{escape(report.note)}[/dim]")


CHAT_SYSTEM_PROMPT = (
    "You are Wanderlust, a friendly and practical travel assistant. "
    "Answer concisely. If you are not sure about a fact such as prices or opening "
    "hours, say so instead of guessing."
)


@app.command()
def chat(
    message: Annotated[str, typer.Argument(help='What to ask, e.g. "Is Lisbon hilly?"')],
) -> None:
    """Ask the configured language model a question (a quick check that your setup works).

    Answers here are not grounded in the travel guides yet; that comes with 'wanderlust ask'.
    """
    settings = _load_settings()
    try:
        provider = build_provider(settings)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    conversation = Conversation(system=CHAT_SYSTEM_PROMPT).add_user(message)
    console.print(f"[dim]{escape(provider.name)} · {escape(provider.model)}[/dim]")
    wrapper = _wrapper()
    try:
        for piece in provider.stream(conversation, max_tokens=4000):
            wrapper.feed(piece)
        wrapper.close()
    except LLMError as exc:
        console.print(f"\n[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    console.print()


@app.command()
def plan(
    request: Annotated[
        str,
        typer.Argument(help='What you want, e.g. "4 days in Lisbon in March, food and history".'),
    ],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show each tool call as the agent works.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the itinerary as JSON instead of a table.")
    ] = False,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=2, max=40, help="Most model calls the agent may make."),
    ] = DEFAULT_MAX_STEPS,
    mode: Annotated[
        SearchMode, typer.Option("--mode", "-m", help="Guide search: hybrid, keyword or vector.")
    ] = SearchMode.HYBRID,
) -> None:
    """Plan a trip: a day-by-day itinerary in which every stop cites its source.

    The agent searches the travel guides, checks the weather and distances, then writes
    the plan. Use --verbose to watch the tool calls.
    """
    settings = _load_settings()
    embedder = _embedder_or_exit(settings)
    try:
        provider = build_provider(settings)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    # With --json, stdout is only the JSON; progress goes to stderr.
    out = Console(stderr=True) if as_json else console
    loader = build_loader(settings)
    client = build_tools_client(settings.data_dir)
    status = out.status("Planning your trip...")

    def on_build(destination: str) -> None:
        status.update(f"Building the {escape(destination)} guide index (first time only)...")

    def on_event(event: AgentEvent) -> None:
        if isinstance(event, ModelReplied):
            if verbose and event.response.text.strip():
                out.print(f"[dim italic]{escape(event.response.text.strip())}[/dim italic]")
            status.update("Thinking...")
        elif isinstance(event, ToolStarted):
            status.update(escape(describe_call(event.call)) + "...")
        elif verbose:
            print_step(out, event.call, event.result)

    try:
        library = GuideLibrary(
            index_opener(
                IndexStore(settings.data_dir),
                loader,
                embedder,
                language=settings.wikivoyage_language,
                on_build=on_build,
            )
        )
        agent = PlanningAgent(
            provider,
            library,
            travel_tools(client, language=settings.wikivoyage_language),
            mode=mode,
            max_steps=max_steps,
            today=lambda: client.clock().date(),
            on_event=on_event,
        )
        out.print(
            f"[dim]{escape(provider.name)} · {escape(provider.model)} · "
            f"planning: {escape(request)}[/dim]",
            highlight=False,
        )
        with status:
            result = agent.plan(request)
    except PlanningError as exc:
        out.print(f"[red]{escape(str(exc))}[/red]")
        if exc.model_text.strip():
            out.print(f"[dim]The model said: {escape(exc.model_text.strip())}[/dim]")
        out.print(f"[dim]{escape(progress_summary(exc.progress))}.[/dim]")
        raise typer.Exit(code=1) from exc
    except (LLMError, ValueError) as exc:
        out.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        loader.close()
        client.close()

    if as_json:
        typer.echo(plan_json(result))
    else:
        print_itinerary(console, result)


if __name__ == "__main__":  # pragma: no cover
    app()
