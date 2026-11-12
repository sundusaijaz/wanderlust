"""Number retrieved passages so an answer can cite them as [1], [2], ..."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from wanderlust.retrieval.chunking import Chunk
from wanderlust.retrieval.search import SearchResult


@dataclass(frozen=True)
class Source:
    """One numbered passage given to the model."""

    number: int
    chunk: Chunk

    @property
    def label(self) -> str:
        """Human-readable name, e.g. 'Lisbon > Eat · Pastéis de Belém'."""
        if self.chunk.listing_name:
            return f"{self.chunk.breadcrumb} · {self.chunk.listing_name}"
        return self.chunk.breadcrumb

    @property
    def url(self) -> str:
        return self.chunk.attribution.url


def number_sources(results: Sequence[SearchResult]) -> list[Source]:
    """Give each distinct passage a number, starting at 1, in ranking order."""
    sources: list[Source] = []
    seen: set[str] = set()
    for result in results:
        if result.chunk.id in seen:
            continue
        seen.add(result.chunk.id)
        sources.append(Source(number=len(sources) + 1, chunk=result.chunk))
    return sources


def format_sources(sources: Sequence[Source]) -> str:
    """Render sources as tagged blocks for the prompt.

    Each passage is wrapped in its own tag with its number and location, so the
    model can tell passages apart and cite them precisely.
    """
    blocks = []
    for source in sources:
        blocks.append(
            f'<passage number="{source.number}" location="{_attr(source.label)}">\n'
            f"{source.chunk.text}\n"
            f"</passage>"
        )
    return "\n".join(blocks)


def _attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
