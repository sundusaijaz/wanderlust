"""Answer questions from a destination's guide, citing the passages used."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from wanderlust.llm.types import Conversation, LLMProvider, LLMResponse, StopReason
from wanderlust.qa.citations import CitationReport, check_citations
from wanderlust.qa.prompt import NOT_COVERED, build_conversation, is_not_covered
from wanderlust.qa.sources import Source, number_sources
from wanderlust.retrieval.search import Searcher, SearchMode, SearchResponse

DEFAULT_PASSAGES = 6
DEFAULT_MAX_TOKENS = 4000


@dataclass(frozen=True)
class PreparedQuestion:
    """A question with its retrieved passages, ready to send to the model."""

    question: str
    destination: str
    search: SearchResponse
    sources: list[Source]
    conversation: Conversation | None

    @property
    def has_context(self) -> bool:
        return bool(self.sources)


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: list[Source]
    citations: CitationReport
    covered: bool
    """False when the guide had nothing relevant or the model said so."""
    refused: bool = False
    response: LLMResponse | None = None


class GroundedAnswerer:
    """Retrieve passages for a question, ask the model, and check its citations."""

    def __init__(
        self,
        searcher: Searcher,
        provider: LLMProvider,
        *,
        passages: int = DEFAULT_PASSAGES,
        mode: SearchMode | str = SearchMode.HYBRID,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.searcher = searcher
        self.provider = provider
        self.passages = passages
        self.mode = SearchMode(mode)
        self.max_tokens = max_tokens

    @property
    def destination(self) -> str:
        return self.searcher.index.meta.destination

    def prepare(self, question: str) -> PreparedQuestion:
        search = self.searcher.search(question, k=self.passages, mode=self.mode)
        sources = number_sources(search.results)
        conversation = build_conversation(question, self.destination, sources) if sources else None
        return PreparedQuestion(
            question=question,
            destination=self.destination,
            search=search,
            sources=sources,
            conversation=conversation,
        )

    def answer(self, question: str) -> Answer:
        """Answer in one call (no streaming)."""
        prepared = self.prepare(question)
        if prepared.conversation is None:
            return self._nothing_found(prepared)
        response = self.provider.complete(prepared.conversation, max_tokens=self.max_tokens)
        return self.finish(prepared, response.text, response=response)

    def stream(self, prepared: PreparedQuestion) -> Iterator[str]:
        """Stream the answer text for a prepared question that has passages."""
        if prepared.conversation is None:
            raise ValueError("nothing was retrieved for this question; nothing to send")
        yield from self.provider.stream(prepared.conversation, max_tokens=self.max_tokens)

    def finish(
        self, prepared: PreparedQuestion, text: str, *, response: LLMResponse | None = None
    ) -> Answer:
        """Build the final answer from the model's text and check its citations."""
        refused = response is not None and response.stop_reason is StopReason.REFUSAL
        return Answer(
            question=prepared.question,
            text=text,
            sources=prepared.sources,
            citations=check_citations(text, prepared.sources),
            covered=not refused and not is_not_covered(text),
            refused=refused,
            response=response,
        )

    def _nothing_found(self, prepared: PreparedQuestion) -> Answer:
        text = f"{NOT_COVERED}: no passages in the {prepared.destination} guide match the question."
        return Answer(
            question=prepared.question,
            text=text,
            sources=[],
            citations=check_citations(text, []),
            covered=False,
        )
