"""Wire the planning agent to the guides, the travel tools and a model."""

from __future__ import annotations

from collections.abc import Callable

from wanderlust.agent.guides import GuideLibrary, index_opener
from wanderlust.agent.planner import DEFAULT_MAX_STEPS, AgentEvent, PlanningAgent
from wanderlust.config import Settings
from wanderlust.llm.types import LLMProvider
from wanderlust.retrieval.embeddings import Embedder
from wanderlust.retrieval.index import IndexStore
from wanderlust.retrieval.search import SearchMode
from wanderlust.tools.http import OpenMeteoClient
from wanderlust.tools.registry import travel_tools
from wanderlust.wikivoyage.service import GuideLoader


def build_agent(
    settings: Settings,
    provider: LLMProvider,
    *,
    loader: GuideLoader,
    embedder: Embedder | None,
    tools_client: OpenMeteoClient,
    mode: SearchMode | str = SearchMode.HYBRID,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_event: Callable[[AgentEvent], None] | None = None,
    on_build: Callable[[str], None] | None = None,
) -> PlanningAgent:
    """A planning agent using saved guide indexes (built on first use) and Open-Meteo.

    The caller owns ``loader`` and ``tools_client`` and closes them when done.
    "Today" comes from the tools client's clock, so the agent and the weather
    tool always agree on the date.
    """
    library = GuideLibrary(
        index_opener(
            IndexStore(settings.data_dir),
            loader,
            embedder,
            language=settings.wikivoyage_language,
            on_build=on_build,
        )
    )
    return PlanningAgent(
        provider,
        library,
        travel_tools(tools_client, language=settings.wikivoyage_language),
        mode=mode,
        max_steps=max_steps,
        today=lambda: tools_client.clock().date(),
        on_event=on_event,
    )
