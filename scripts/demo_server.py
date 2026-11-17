"""Run the web interface offline, with a scripted agent and a made-up town.

No API key or internet connection is needed: the guide, the weather and the
model's replies are all stand-ins from the test suite. Useful for trying the
interface and for taking screenshots.

    uv run python scripts/demo_server.py            # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import uvicorn
from render_screenshots import (
    DEMO_PLAN_REQUEST,
    DemoEmbedder,
    DemoPlanner,
    demo_loader,
    demo_tools_client,
)

from wanderlust.config import Settings
from wanderlust.web.app import Services, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--delay", type=float, default=0.8, help="Seconds per agent step.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as data_dir:
        settings = Settings(_env_file=None, data_dir=Path(data_dir))
        services = Services(
            provider=lambda settings: DemoPlanner(delay=args.delay),
            embedder=lambda settings: DemoEmbedder(),
            loader=demo_loader,
            tools_client=demo_tools_client,
        )
        app = create_app(settings, services, demo=True)
        print(f"Try: http://{args.host}:{args.port}/?q={DEMO_PLAN_REQUEST.replace(' ', '+')}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
