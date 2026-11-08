"""Regenerate the terminal screenshots used in the README.

Runs real CLI commands with Rich's recording console and saves each one as an
SVG in docs/assets/. Run it after changing CLI output:

    uv run python scripts/render_screenshots.py
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path

from rich.console import Console

import wanderlust.cli as cli

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

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


def render(commands: list[list[str]], filename: str, title: str) -> Path:
    console = Console(record=True, width=64, force_terminal=True, color_system="truecolor")
    cli.console = console
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
    path = render([["--version"], ["config"]], "cli-config.svg", "wanderlust")
    print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


if __name__ == "__main__":
    main()
