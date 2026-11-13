"""Regenerate the terminal screenshots used in the README.

Runs real CLI commands with Rich's recording console and saves each one as an
SVG in docs/assets/. Run it after changing CLI output:

    uv run python scripts/render_screenshots.py
"""

from __future__ import annotations

import os
import shlex
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path

import httpx
from rich.console import Console
from rich.markup import escape

import wanderlust.cli as cli
from wanderlust.llm.fake import ScriptedProvider
from wanderlust.wikivoyage.cache import ArticleCache
from wanderlust.wikivoyage.client import WikivoyageClient
from wanderlust.wikivoyage.service import GuideLoader

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
# A hand-written answer for the 'ask' screenshot (no API key or model call needed).
# It only states facts from the made-up guide and cites the passages it uses.
DEMO_ANSWER = (
    "Take the heritage tram from the station: it runs to the lighthouse every 15 minutes "
    "[1]. The Old Lighthouse is open 09:00-18:00 and costs 5 coins for adults (children "
    "go free); climb the 120 steps at sunset for the best view [3]. Opening hours and "
    "prices can change, so check before you go."
)

# Made-up guide from the test suite, so screenshots never reproduce Wikivoyage text.
DEMO_ARTICLE = ROOT / "tests" / "fixtures" / "examplestad.wikitext"

# Placeholder values only: the screenshot shows "configured", never the key itself.
DEMO_ENV = {
    "WANDERLUST_LLM_PROVIDER": "anthropic",
    "ANTHROPIC_API_KEY": "placeholder-for-screenshot",
    "WANDERLUST_WIKIVOYAGE_LANGUAGE": "en",
    "WANDERLUST_EMBEDDING_PROVIDER": "local",
}


@contextmanager
def demo_environment():
    """Run inside an empty folder with demo env vars, then restore everything."""
    saved = dict(os.environ)
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        os.environ.update(DEMO_ENV)
        try:
            yield
        finally:
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(saved)


def demo_loader(settings) -> GuideLoader:
    """A guide loader that serves the made-up Examplestad article without network access."""
    wikitext = DEMO_ARTICLE.read_text(encoding="utf-8")
    page = {"title": "Examplestad", "pageid": 1, "revid": 1, "wikitext": wikitext}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"parse": page})

    client = WikivoyageClient(
        "en", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return GuideLoader(client, ArticleCache(settings.data_dir))


class DemoEmbedder:
    """Tiny offline embedder (hashed bag of words) so the search screenshot needs no API."""

    name = "demo:bag-of-words"

    def embed(self, texts):
        import hashlib

        import numpy as np

        from wanderlust.retrieval.embeddings import normalize
        from wanderlust.retrieval.keyword import tokenize

        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % 64
                vectors[row, bucket] += 1.0
        return normalize(vectors)


DEMO_PLACES = {
    "examplestad": {
        "name": "Examplestad",
        "latitude": 12.34,
        "longitude": -7.65,
        "country": "Imaginaria",
        "admin1": "Coast Province",
    },
    "otherville": {
        "name": "Otherville",
        "latitude": 12.5,
        "longitude": -7.5,
        "country": "Imaginaria",
        "admin1": "Coast Province",
    },
}


def demo_tools_client(data_dir):
    """Made-up places and weather, with 'today' fixed so screenshots are reproducible."""
    from datetime import UTC, date, datetime

    from wanderlust.tools.http import OpenMeteoClient

    def api(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if request.url.host.startswith("geocoding"):
            match = DEMO_PLACES.get(params["name"].casefold())
            return httpx.Response(200, json={"results": [match]} if match else {})
        start = date.fromisoformat(params["start_date"])
        end = date.fromisoformat(params["end_date"])
        days = [date.fromordinal(d) for d in range(start.toordinal(), end.toordinal() + 1)]
        pattern = [
            (1, 22.1, 14.2, 0.0, 5),
            (3, 20.4, 13.8, 0.2, 20),
            (63, 17.0, 12.9, 9.4, 90),
            (80, 18.3, 13.1, 2.6, 60),
            (0, 23.0, 14.9, 0.0, 0),
        ]
        rows = [pattern[i % len(pattern)] for i in range(len(days))]
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": [d.isoformat() for d in days],
                    "weather_code": [r[0] for r in rows],
                    "temperature_2m_max": [r[1] for r in rows],
                    "temperature_2m_min": [r[2] for r in rows],
                    "precipitation_sum": [r[3] for r in rows],
                    "precipitation_probability_max": [r[4] for r in rows],
                }
            },
        )

    return OpenMeteoClient(
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
        clock=lambda: datetime(2026, 11, 12, 9, tzinfo=UTC),
    )


class SilentStatus:
    def __enter__(self) -> SilentStatus:
        return self

    def __exit__(self, *exc) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass


# -- a scripted planning run for the 'plan' screenshot ---------------------------

DEMO_PLAN_REQUEST = "3 days in Examplestad from 13 November, lighthouses and seafood"
DEMO_SEARCHES = [
    "lighthouse and sights",
    "fish soup restaurants and pubs",
    "getting around by tram or bicycle",
]


def _stop(time, name, description, *cites, indoor=False):
    return {
        "time_of_day": time,
        "name": name,
        "description": description,
        "cites": cites,
        "indoor": indoor,
    }


# Every fact comes from the made-up Examplestad guide; "cites" names the passages
# by location and is turned into the passage numbers search_guide really returned.
DEMO_ITINERARY = {
    "title": "Three days of lighthouses and fish soup in Examplestad",
    "summary": (
        "A slow harbour-town weekend: the main square and a bike ride on day one, the "
        "lighthouse at sunset on day two, and fish soup indoors when the rain arrives."
    ),
    "days": [
        {
            "title": "Around the harbour",
            "weather": "Mainly clear, 14 to 22 C",
            "stops": [
                _stop(
                    "morning",
                    "Clock Square",
                    "Start in the main square, where the clock always runs five minutes fast.",
                    "See · Clock Square",
                ),
                _stop(
                    "afternoon",
                    "Bike ride from the station",
                    "Rent a bicycle at the station and explore at your own pace.",
                    "Get around > By bicycle",
                ),
                _stop(
                    "evening",
                    "The Salty Mug",
                    "A cosy pub by the pier for the first evening.",
                    "Eat · The Salty Mug",
                    indoor=True,
                ),
            ],
            "getting_around": "Most sights are within walking distance of the harbour.",
        },
        {
            "title": "The lighthouse at sunset",
            "weather": "Overcast, 14 to 20 C",
            "stops": [
                _stop(
                    "afternoon",
                    "Heritage tram",
                    "Ride the single tram line from the station to the lighthouse.",
                    "Get around > By tram",
                ),
                _stop(
                    "evening",
                    "Old Lighthouse",
                    "Climb the 120 steps to the viewpoint at sunset.",
                    "See · Old Lighthouse",
                ),
            ],
            "getting_around": "The tram runs every 15 minutes; walk back along the harbour.",
        },
        {
            "title": "Fish soup on a rainy day",
            "weather": "Rain, 13 to 17 C, 90% chance",
            "stops": [
                _stop(
                    "afternoon",
                    "Harbour Kitchen",
                    "Family-run and known for fish soup, the local speciality.",
                    "Eat · Harbour Kitchen",
                    "Eat",
                    indoor=True,
                ),
            ],
        },
    ],
    "tips": [
        (
            "The tram runs every 15 minutes, so there is no timetable to plan around",
            "Get around > By tram",
        ),
        ("Pack a windproof layer: winters here are windy", "Understand"),
    ],
    "assumptions": ["Dates taken as Friday 13 to Sunday 15 November 2026."],
}


class DemoPlanner(ScriptedProvider):
    """Researches with a fixed set of tool calls, then submits DEMO_ITINERARY."""

    def __init__(self) -> None:
        from wanderlust.llm.types import ToolCall

        def call(n, name, **arguments):
            return ToolCall(id=f"call-{n}", name=name, arguments=arguments)

        research = [
            ScriptedProvider.tool_request(
                *[
                    call(n, "search_guide", destination="Examplestad", query=query)
                    for n, query in enumerate(DEMO_SEARCHES, start=1)
                ],
                text="I'll research the Examplestad guide for sights, food and transport.",
            ),
            ScriptedProvider.tool_request(
                call(4, "geocode", place="Examplestad"),
                text="Now the weather for the trip dates.",
            ),
            ScriptedProvider.tool_request(
                call(
                    5,
                    "get_weather",
                    latitude=12.34,
                    longitude=-7.65,
                    start_date="2026-11-13",
                    end_date="2026-11-15",
                ),
            ),
        ]
        super().__init__(research, model="demo-model")

    def complete(self, conversation, *, tools=None, max_tokens=16000):
        if self.remaining:
            return super().complete(conversation, tools=tools, max_tokens=max_tokens)
        from wanderlust.llm.types import ToolCall

        self.requests.append((conversation, list(tools or [])))
        return ScriptedProvider.tool_request(
            ToolCall(id="call-6", name="submit_itinerary", arguments=self._itinerary(conversation))
        )

    @staticmethod
    def _itinerary(conversation) -> dict:
        import re
        from datetime import date, timedelta

        from wanderlust.llm.types import ToolResultsTurn

        numbers: dict[str, int] = {}
        for turn in conversation.turns:
            if isinstance(turn, ToolResultsTurn):
                for result in turn.results:
                    for number, location in re.findall(
                        r'<passage number="(\d+)" location="Examplestad > ([^"]+)"',
                        result.content,
                    ):
                        numbers[location] = int(number)

        def cite(location: str) -> int:
            if location not in numbers:
                raise RuntimeError(f"demo searches never returned '{location}'")
            return numbers[location]

        start = date(2026, 11, 13)
        days = []
        for n, day in enumerate(DEMO_ITINERARY["days"]):
            stops = [
                {
                    **{k: v for k, v in stop.items() if k != "cites"},
                    "sources": [cite(c) for c in stop["cites"]],
                }
                for stop in day["stops"]
            ]
            days.append(
                {
                    **day,
                    "day": n + 1,
                    "date": (start + timedelta(days=n)).isoformat(),
                    "base": "Examplestad",
                    "stops": stops,
                }
            )
        tips = [f"{text} [{cite(where)}]." for text, where in DEMO_ITINERARY["tips"]]
        return {**DEMO_ITINERARY, "days": days, "tips": tips}


REAL_BUILD_EMBEDDER = cli.build_embedder


def render(
    commands: list[list[str]],
    filename: str,
    title: str,
    *,
    width: int = 64,
    embedder_factory=None,
    provider_factory=None,
) -> Path:
    console = Console(record=True, width=width, force_terminal=True, color_system="truecolor")
    # Spinners don't belong in a still image: swap in a silent stand-in.
    console.status = lambda *args, **kwargs: SilentStatus()
    cli.console = console
    cli.build_loader = demo_loader
    cli.build_embedder = embedder_factory or REAL_BUILD_EMBEDDER
    cli.build_provider = provider_factory or (
        lambda settings: ScriptedProvider([DEMO_ANSWER], model="demo-model")
    )
    cli.build_tools_client = demo_tools_client
    with demo_environment():
        for args in commands:
            console.print(
                f"[bold green]$[/] wanderlust {escape(shlex.join(args))}", highlight=False
            )
            with suppress(SystemExit):
                cli.app(args, standalone_mode=False)
            console.print()
    out = ASSETS / filename
    console.save_svg(str(out), title=title)
    return out


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    paths = [
        render([["--version"], ["config"]], "cli-config.svg", "wanderlust"),
        render([["guide", "Examplestad"]], "cli-guide.svg", "wanderlust guide", width=96),
        render(
            [["guide", "Examplestad", "--section", "See"]],
            "cli-guide-section.svg",
            "wanderlust guide --section",
            width=96,
        ),
        render(
            [
                ["index", "Examplestad"],
                ["search", "Examplestad", "lighthouse at sunset", "-k", "2"],
            ],
            "cli-search.svg",
            "wanderlust search",
            width=96,
            embedder_factory=lambda settings: DemoEmbedder(),
        ),
        render(
            [
                [
                    "ask",
                    "Examplestad",
                    "How do I get to the lighthouse, and when is it open?",
                    "-k",
                    "3",
                ]
            ],
            "cli-ask.svg",
            "wanderlust ask",
            width=96,
            embedder_factory=lambda settings: DemoEmbedder(),
        ),
        render(
            [["weather", "Examplestad", "--days", "5"], ["distance", "Examplestad", "Otherville"]],
            "cli-weather.svg",
            "wanderlust weather",
            width=96,
        ),
        render(
            [["plan", DEMO_PLAN_REQUEST, "--verbose"]],
            "cli-plan.svg",
            "wanderlust plan",
            width=96,
            embedder_factory=lambda settings: DemoEmbedder(),
            provider_factory=lambda settings: DemoPlanner(),
        ),
    ]
    for path in paths:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
