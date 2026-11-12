import pytest

from wanderlust.llm.fake import ScriptedProvider
from wanderlust.llm.types import LLMResponse, StopReason
from wanderlust.qa.answerer import GroundedAnswerer
from wanderlust.qa.prompt import SYSTEM_PROMPT


def test_answer_cites_retrieved_passages(searcher):
    llm = ScriptedProvider(["Try the fish soup at Harbour Kitchen [1]. Prices may change [1][7]."])
    answerer = GroundedAnswerer(searcher, llm, passages=3)

    answer = answerer.answer("Where can I eat fish soup?")

    assert answer.covered and not answer.refused
    assert 1 <= len(answer.sources) <= 3
    assert [s.number for s in answer.citations.cited] == [1]
    assert answer.citations.unknown == [7]
    assert "fish soup" in answer.sources[0].chunk.text.lower()

    conversation, tools = llm.requests[0]
    assert conversation.system == SYSTEM_PROMPT
    assert "Question: Where can I eat fish soup?" in conversation.turns[0].text
    assert tools == []


def test_not_covered_answer_is_detected(searcher):
    llm = ScriptedProvider(["The guide doesn't cover this. It does mention a tram [2]."])
    answer = GroundedAnswerer(searcher, llm).answer("Is there a casino in the lighthouse?")
    assert not answer.covered
    assert [s.number for s in answer.citations.cited] == [2]


def test_nothing_retrieved_skips_the_model(searcher):
    llm = ScriptedProvider([])
    answer = GroundedAnswerer(searcher, llm, mode="keyword").answer("snorkelling")
    assert not answer.covered
    assert answer.sources == []
    assert "no passages in the Examplestad guide" in answer.text
    assert llm.requests == []


def test_refusal_is_reported(searcher):
    refusal = LLMResponse(text="", stop_reason=StopReason.REFUSAL, provider="fake", model="m")
    answer = GroundedAnswerer(searcher, ScriptedProvider([refusal])).answer("tram")
    assert answer.refused and not answer.covered


def test_streaming_flow(searcher):
    llm = ScriptedProvider(["The tram runs every 15 minutes [1]."])
    answerer = GroundedAnswerer(searcher, llm)

    prepared = answerer.prepare("How often does the tram run?")
    assert prepared.has_context
    assert prepared.destination == "Examplestad"

    text = "".join(answerer.stream(prepared))
    answer = answerer.finish(prepared, text)
    assert answer.text == "The tram runs every 15 minutes [1]."
    assert answer.covered
    assert answer.citations.cited[0].number == 1


def test_streaming_without_context_is_an_error(searcher):
    answerer = GroundedAnswerer(searcher, ScriptedProvider([]), mode="keyword")
    prepared = answerer.prepare("snorkelling")
    assert not prepared.has_context
    with pytest.raises(ValueError, match="nothing was retrieved"):
        list(answerer.stream(prepared))


def test_passage_count_and_mode_are_used(searcher):
    answerer = GroundedAnswerer(searcher, ScriptedProvider([]), passages=2, mode="keyword")
    prepared = answerer.prepare("the town harbour lighthouse tram")
    assert len(prepared.sources) <= 2
    assert prepared.search.mode == "keyword"
