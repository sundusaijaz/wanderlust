"""The web API: service status and a planning endpoint that streams the agent's work."""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from wanderlust import __version__
from wanderlust.agent.describe import (
    describe_call,
    plan_payload,
    progress_summary,
    summarize_result,
)
from wanderlust.agent.planner import (
    DEFAULT_MAX_STEPS,
    AgentEvent,
    ModelReplied,
    PlanningError,
    ToolStarted,
)
from wanderlust.agent.setup import build_agent
from wanderlust.config import MissingAPIKeyError, Settings
from wanderlust.llm.factory import build_provider
from wanderlust.llm.types import LLMError, LLMProvider
from wanderlust.retrieval.embeddings import Embedder, EmbeddingError, build_embedder
from wanderlust.tools.http import OpenMeteoClient
from wanderlust.tools.registry import build_tools_client
from wanderlust.wikivoyage.service import GuideLoader, build_loader

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
KEEPALIVE_SECONDS = 15.0
DEFAULT_MAX_CONCURRENT_PLANS = 2


@dataclass(frozen=True)
class Services:
    """Where the app gets its collaborators. Tests and the offline demo swap these."""

    provider: Callable[[Settings], LLMProvider] = build_provider
    embedder: Callable[[Settings], Embedder | None] = build_embedder
    loader: Callable[[Settings], GuideLoader] = build_loader
    tools_client: Callable[[Path], OpenMeteoClient] = build_tools_client


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=3, max_length=500)
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=2, le=30)


class _Cancelled(Exception):
    """The browser went away; stop planning."""


_DONE = object()


def sse(event: str, data: dict[str, Any]) -> str:
    """One server-sent event. JSON keeps the data on a single line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    *,
    demo: bool = False,
    max_concurrent_plans: int = DEFAULT_MAX_CONCURRENT_PLANS,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
) -> FastAPI:
    """Build the app. ``demo`` marks an offline demo with scripted answers."""
    settings = settings or Settings()
    services = services or Services()
    slots = threading.BoundedSemaphore(max_concurrent_plans)

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

    @app.post("/api/plan")
    def plan(body: PlanRequest) -> StreamingResponse:
        """Plan a trip, streaming the agent's progress as server-sent events."""
        if not slots.acquire(blocking=False):
            raise HTTPException(
                status_code=429, detail="The planner is busy. Please try again in a minute."
            )
        try:
            events = _plan_events(body, settings, services, keepalive_seconds)
        except BaseException:
            slots.release()
            raise

        def stream() -> Iterator[str]:
            try:
                yield from events
            finally:
                slots.release()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _plan_events(
    body: PlanRequest, settings: Settings, services: Services, keepalive_seconds: float
) -> Iterator[str]:
    """Run the agent in a worker thread and yield its progress as events."""
    events: queue.Queue[object] = queue.Queue()
    cancelled = threading.Event()

    def emit(event: str, data: dict[str, Any]) -> None:
        events.put(sse(event, data))

    def on_event(event: AgentEvent) -> None:
        if cancelled.is_set():
            raise _Cancelled
        if isinstance(event, ModelReplied):
            if event.response.text.strip():
                emit("thinking", {"text": event.response.text.strip()})
        elif isinstance(event, ToolStarted):
            call = event.call
            emit("tool_started", {"id": call.id, "name": call.name, "label": describe_call(call)})
        else:
            call, result = event.call, event.result
            emit(
                "tool_finished",
                {
                    "id": call.id,
                    "name": call.name,
                    "summary": summarize_result(call, result),
                    "is_error": result.is_error,
                },
            )

    def on_build(destination: str) -> None:
        emit("status", {"message": f"Building the {destination} guide index (first time only)"})

    def run() -> None:
        loader = client = None
        try:
            provider = services.provider(settings)
            embedder = services.embedder(settings)
            loader = services.loader(settings)
            client = services.tools_client(settings.data_dir)
            agent = build_agent(
                settings,
                provider,
                loader=loader,
                embedder=embedder,
                tools_client=client,
                max_steps=body.max_steps,
                on_event=on_event,
                on_build=on_build,
            )
            emit("start", {"provider": provider.name, "model": provider.model})
            result = agent.plan(body.request)
            emit(
                "plan",
                {
                    **plan_payload(result),
                    "summary": progress_summary(result.progress),
                    "used_weather": result.used_weather,
                },
            )
        except _Cancelled:
            logger.info("Planning cancelled: the client disconnected")
        except PlanningError as exc:
            emit(
                "error",
                {
                    "message": str(exc),
                    "model_text": exc.model_text.strip(),
                    "summary": progress_summary(exc.progress),
                },
            )
        except (MissingAPIKeyError, EmbeddingError, LLMError, ValueError) as exc:
            emit("error", {"message": str(exc)})
        except Exception:
            logger.exception("Planning failed")
            emit("error", {"message": "Something went wrong while planning. Please try again."})
        finally:
            for resource in (loader, client):
                if resource is not None:
                    resource.close()
            events.put(_DONE)

    worker = threading.Thread(target=run, name="wanderlust-plan", daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = events.get(timeout=keepalive_seconds)
            except queue.Empty:
                yield ": keep-alive\n\n"  # a comment line keeps proxies from timing out
                continue
            if item is _DONE:
                yield sse("done", {})
                return
            assert isinstance(item, str)
            yield item
    finally:
        cancelled.set()
