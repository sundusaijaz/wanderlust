"""Keyword, vector and hybrid search over a guide index."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from wanderlust.retrieval.chunking import Chunk
from wanderlust.retrieval.embeddings import Embedder
from wanderlust.retrieval.index import GuideIndex

RRF_K = 60  # standard constant from Cormack et al. (2009)


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    VECTOR = "vector"


class SearchError(RuntimeError):
    """The requested search cannot be run with this index and configuration."""


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None


@dataclass(frozen=True)
class SearchResponse:
    results: list[SearchResult]
    mode: SearchMode
    note: str | None = None


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = RRF_K
) -> list[tuple[int, float]]:
    """Merge ranked lists: each item scores ``sum(1 / (k + rank))`` over the lists it is in.

    RRF only uses ranks, so BM25 scores and cosine similarities never have to
    be put on the same scale.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class Searcher:
    """Runs queries against one :class:`GuideIndex`."""

    def __init__(
        self, index: GuideIndex, embedder: Embedder | None = None, *, candidates: int = 50
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.candidates = candidates

    def vector_unavailable_reason(self) -> str | None:
        """Why vector search can't run, or ``None`` if it can."""
        built_with = self.index.meta.embedder
        if self.index.vector_index is None or built_with is None:
            return "this index was built without embeddings"
        if self.embedder is None:
            return "no embedding provider is configured"
        if self.embedder.name != built_with:
            return (
                f"the index was built with {built_with} but {self.embedder.name} is "
                "configured; rebuild it with 'wanderlust index --refresh'"
            )
        return None

    def search(
        self, query: str, *, k: int = 5, mode: SearchMode | str = SearchMode.HYBRID
    ) -> SearchResponse:
        # Normalise first: the checks below compare enum members by identity.
        mode = SearchMode(mode)
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")

        note = None
        if mode is not SearchMode.KEYWORD:
            reason = self.vector_unavailable_reason()
            if reason and mode is SearchMode.VECTOR:
                raise SearchError(f"Vector search is unavailable: {reason}.")
            if reason:
                mode, note = SearchMode.KEYWORD, f"Using keyword search only: {reason}."

        keyword: list[tuple[int, float]] = []
        vector: list[tuple[int, float]] = []
        if mode is not SearchMode.VECTOR:
            keyword = self.index.bm25.search(query, self.candidates)
        if mode is not SearchMode.KEYWORD:
            assert self.embedder is not None and self.index.vector_index is not None
            query_vector = self.embedder.embed([query])[0]
            vector = self.index.vector_index.search(query_vector, self.candidates)

        keyword_ranks = {i: rank for rank, (i, _) in enumerate(keyword, start=1)}
        vector_ranks = {i: rank for rank, (i, _) in enumerate(vector, start=1)}

        if mode is SearchMode.HYBRID:
            ranked = reciprocal_rank_fusion([[i for i, _ in keyword], [i for i, _ in vector]])
        else:
            ranked = keyword or vector

        results = [
            SearchResult(
                chunk=self.index.chunks[i],
                score=score,
                keyword_rank=keyword_ranks.get(i),
                vector_rank=vector_ranks.get(i),
            )
            for i, score in ranked[:k]
        ]
        return SearchResponse(results=results, mode=mode, note=note)
