"""Let the planner search any destination's guide, with stable passage numbers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from wanderlust.qa.sources import Source, format_sources
from wanderlust.retrieval.builder import build_index
from wanderlust.retrieval.chunking import Chunk
from wanderlust.retrieval.embeddings import Embedder, EmbeddingError
from wanderlust.retrieval.index import IndexStore
from wanderlust.retrieval.search import Searcher, SearchError, SearchMode
from wanderlust.tools.http import ToolError
from wanderlust.tools.registry import Tool, ToolRegistry
from wanderlust.wikivoyage.client import ArticleNotFoundError, WikivoyageError
from wanderlust.wikivoyage.service import GuideLoader

DEFAULT_MAX_GUIDES = 4
DEFAULT_PASSAGES = 5


class SourceBook:
    """Numbers passages across every search in one planning run.

    A passage keeps its number when it turns up again, so the model can cite
    [3] from its first search after five more searches.
    """

    def __init__(self) -> None:
        self._by_chunk: dict[str, Source] = {}
        self._by_number: dict[int, Source] = {}

    def add(self, chunk: Chunk) -> Source:
        source = self._by_chunk.get(chunk.id)
        if source is None:
            source = Source(number=len(self._by_chunk) + 1, chunk=chunk)
            self._by_chunk[chunk.id] = source
            self._by_number[source.number] = source
        return source

    def get(self, number: int) -> Source | None:
        return self._by_number.get(number)

    def __contains__(self, number: object) -> bool:
        return number in self._by_number

    def __len__(self) -> int:
        return len(self._by_number)

    @property
    def sources(self) -> list[Source]:
        return list(self._by_number.values())

    def select(self, numbers: Iterable[int]) -> list[Source]:
        """The known sources among ``numbers``, in the given order."""
        return [self._by_number[n] for n in numbers if n in self._by_number]


SearcherOpener = Callable[[str], Searcher]


class GuideLibrary:
    """Opens destination guides on demand and keeps them open for the run.

    ``max_guides`` caps how many different guides one plan may open, so a
    confused model can't download half of Wikivoyage.
    """

    def __init__(self, opener: SearcherOpener, *, max_guides: int = DEFAULT_MAX_GUIDES) -> None:
        if max_guides < 1:
            raise ValueError("max_guides must be at least 1")
        self._opener = opener
        self.max_guides = max_guides
        self._open: dict[str, Searcher] = {}

    @property
    def destinations(self) -> list[str]:
        """Guides opened so far, by their real titles."""
        names: dict[str, None] = {}
        for searcher in self._open.values():
            names.setdefault(searcher.index.meta.destination, None)
        return list(names)

    def searcher(self, destination: str) -> Searcher:
        key = _key(destination)
        if key in self._open:
            return self._open[key]
        if len(self.destinations) >= self.max_guides:
            raise ToolError(
                f"This plan already uses {self.max_guides} guides "
                f"({', '.join(self.destinations)}); search one of those instead."
            )
        try:
            searcher = self._opener(destination)
        except ArticleNotFoundError as exc:
            raise ToolError(
                f"There is no Wikivoyage guide called '{destination}'."
                + (f" Did you mean '{exc.suggestion}'?" if exc.suggestion else "")
            ) from exc
        except (WikivoyageError, EmbeddingError) as exc:
            raise ToolError(f"Could not open the {destination} guide: {exc}") from exc
        # 'Lisboa' redirects to 'Lisbon': keep using the guide that is already open.
        searcher = self._open.setdefault(_key(searcher.index.meta.destination), searcher)
        self._open[key] = searcher
        return searcher


def _key(name: str) -> str:
    return " ".join(name.replace("_", " ").split()).casefold()


def index_opener(
    store: IndexStore,
    loader: GuideLoader,
    embedder: Embedder | None,
    *,
    language: str = "en",
    on_build: Callable[[str], None] | None = None,
) -> SearcherOpener:
    """Open a saved index, or build (and save) it the first time a guide is needed."""

    def open_guide(destination: str) -> Searcher:
        index = store.load(language, destination)
        if index is None:
            if on_build is not None:
                on_build(destination)
            index = build_index(destination, loader, embedder).index
            store.save(index)
        return Searcher(index, embedder)

    return open_guide


class GuideSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination: str = Field(
        min_length=2,
        description="The Wikivoyage guide to search: a city, region or country, e.g. 'Lisbon'.",
    )
    query: str = Field(
        min_length=2, description="What to look for, e.g. 'seafood restaurants' or 'day hikes'."
    )
    limit: int = Field(default=DEFAULT_PASSAGES, ge=1, le=10, description="Passages to return.")


def guide_tools(
    library: GuideLibrary, book: SourceBook, *, mode: SearchMode | str = SearchMode.HYBRID
) -> ToolRegistry:
    """The ``search_guide`` tool, numbering every passage it returns in ``book``."""
    search_mode = SearchMode(mode)

    def search_guide(args: GuideSearchArgs) -> Any:
        searcher = library.searcher(args.destination)
        destination = searcher.index.meta.destination
        try:
            response = searcher.search(args.query, k=args.limit, mode=search_mode)
        except SearchError as exc:
            raise ToolError(str(exc)) from exc
        if not response.results:
            return (
                f"No passages in the {destination} guide match '{args.query}'. "
                "Try other words, or plan around what the guide does cover."
            )
        sources = [book.add(result.chunk) for result in response.results]
        return f"Passages from the {destination} guide:\n{format_sources(sources)}"

    return ToolRegistry(
        [
            Tool(
                name="search_guide",
                description=(
                    "Search a destination's Wikivoyage travel guide. Returns numbered "
                    "passages; cite them by number. Passages about a listing (a sight, "
                    "restaurant or hotel) include its coordinates when known."
                ),
                args_model=GuideSearchArgs,
                handler=search_guide,
            )
        ]
    )
