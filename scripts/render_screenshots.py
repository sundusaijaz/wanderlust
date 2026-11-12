"""Regenerate the terminal screenshots used in the README.

Runs real CLI commands with Rich's recording console and saves each one as an
SVG in docs/assets/. Run it after changing CLI output:

    uv run python scripts/render_screenshots.py
"""

from __future__ import annotations

import os
import shlex
import tempfile
from contextlib import contextmanager, nullcontext, suppress
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


class SilentStatus:
    def update(self, *args, **kwargs) -> None:
        pass


REAL_BUILD_EMBEDDER = cli.build_embedder


def render(
    commands: list[list[str]],
    filename: str,
    title: str,
    *,
    width: int = 64,
    embedder_factory=None,
) -> Path:
    console = Console(record=True, width=width, force_terminal=True, color_system="truecolor")
    # Spinners don't belong in a still image: swap in a silent stand-in.
    console.status = lambda *args, **kwargs: nullcontext(SilentStatus())
    cli.console = console
    cli.build_loader = demo_loader
    cli.build_embedder = embedder_factory or REAL_BUILD_EMBEDDER
    cli.build_provider = lambda settings: ScriptedProvider([DEMO_ANSWER], model="demo-model")
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
    ]
    for path in paths:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
