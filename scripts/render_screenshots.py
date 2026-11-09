"""Regenerate the terminal screenshots used in the README.

Runs real CLI commands with Rich's recording console and saves each one as an
SVG in docs/assets/. Run it after changing CLI output:

    uv run python scripts/render_screenshots.py
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path

import httpx
from rich.console import Console

import wanderlust.cli as cli
from wanderlust.wikivoyage.cache import ArticleCache
from wanderlust.wikivoyage.client import WikivoyageClient
from wanderlust.wikivoyage.service import GuideLoader

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
# Made-up guide from the test suite, so screenshots never reproduce Wikivoyage text.
DEMO_ARTICLE = ROOT / "tests" / "fixtures" / "examplestad.wikitext"

# Placeholder values only: the screenshot shows "configured", never the key itself.
DEMO_ENV = {
    "WANDERLUST_LLM_PROVIDER": "anthropic",
    "ANTHROPIC_API_KEY": "placeholder-for-screenshot",
    "WANDERLUST_WIKIVOYAGE_LANGUAGE": "en",
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


def render(commands: list[list[str]], filename: str, title: str, *, width: int = 64) -> Path:
    console = Console(record=True, width=width, force_terminal=True, color_system="truecolor")
    console.status = lambda *args, **kwargs: nullcontext()  # spinners don't belong in a still
    cli.console = console
    cli.build_loader = demo_loader
    with demo_environment():
        for args in commands:
            console.print(f"[bold green]$[/] wanderlust {' '.join(args)}")
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
    ]
    for path in paths:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
