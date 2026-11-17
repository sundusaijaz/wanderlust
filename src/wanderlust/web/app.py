"""The web API behind the planner page."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from wanderlust import __version__
from wanderlust.config import MissingAPIKeyError, Settings
from wanderlust.llm.factory import build_provider
from wanderlust.llm.types import LLMProvider
from wanderlust.retrieval.embeddings import Embedder, build_embedder
from wanderlust.tools.http import OpenMeteoClient
from wanderlust.tools.registry import build_tools_client
from wanderlust.wikivoyage.service import GuideLoader, build_loader

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@dataclass(frozen=True)
class Services:
    """Where the app gets its collaborators. Tests and the offline demo swap these."""

    provider: Callable[[Settings], LLMProvider] = build_provider
    embedder: Callable[[Settings], Embedder | None] = build_embedder
    loader: Callable[[Settings], GuideLoader] = build_loader
    tools_client: Callable[[Path], OpenMeteoClient] = build_tools_client


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    *,
    demo: bool = False,
) -> FastAPI:
    """Build the app. ``demo`` marks an offline demo with scripted answers."""
    settings = settings or Settings()
    services = services or Services()

    app = FastAPI(title="Wanderlust", version=__version__, docs_url=None, redoc_url=None)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        """Which model is configured, and whether planning can run."""
        problem = None
        provider_name, model = settings.llm_provider.value, None
        try:
            provider = services.provider(settings)
            provider_name, model = provider.name, provider.model
        except MissingAPIKeyError as exc:
            problem = str(exc)
        return {
            "version": __version__,
            "provider": provider_name,
            "model": model,
            "ready": problem is None,
            "problem": problem,
            "demo": demo,
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
