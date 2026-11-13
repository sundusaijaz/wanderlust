from datetime import date

import pytest
from pydantic import BaseModel

from wanderlust.agent.planner import (
    SUBMIT_SPEC,
    ModelReplied,
    PlanningAgent,
    PlanningError,
    ToolFinished,
    ToolStarted,
)
from wanderlust.llm.fake import ScriptedProvider
from wanderlust.llm.types import (
    LLMError,
    LLMResponse,
    StopReason,
    ToolCall,
    ToolResultsTurn,
    Usage,
    UserTurn,
)
from wanderlust.tools.registry import Tool, ToolRegistry

TODAY = date(2026, 11, 13)


def call(name, n=1, **arguments):
    return ToolCall(id=f"{name}-{n}", name=name, arguments=arguments)


def search(query="lighthouse at sunset", n=1, limit=3):
    return call("search_guide", n, destination="Examplestad", query=query, limit=limit)


def itinerary(*sources, **extra):
    return {
        "title": "A day in Examplestad",
        "summary": "Lighthouse and fish soup.",
        "days": [
            {
                "day": 1,
                "base": "Examplestad",
                "title": "Harbour and lighthouse",
                "stops": [
                    {
                        "time_of_day": "evening",
                        "name": "Old Lighthouse",
                        "description": "Climb the 120 steps at sunset.",
                        "sources": list(sources or (1,)),
                    }
                ],
            }
        ],
        **extra,
    }


def submit(*sources, n=1, **extra):
    return call("submit_itinerary", n, **itinerary(*sources, **extra))


USAGE = Usage(10, 5)


def reply(*calls, text="", usage=USAGE):
    response = ScriptedProvider.tool_request(*calls, text=text)
    return LLMResponse(
        text=response.text,
        stop_reason=response.stop_reason,
        provider="fake",
        model="scripted",
        tool_calls=response.tool_calls,
        usage=usage,
    )


class WeatherArgs(BaseModel):
    place: str


@pytest.fixture
def weather_calls():
    return []


@pytest.fixture
def travel(weather_calls):
    def get_weather(args):
        weather_calls.append(args.place)
        return {"kind": "forecast", "days": [{"summary": "Clear sky"}]}

    return ToolRegistry(
        [Tool(name="get_weather", description="w", args_model=WeatherArgs, handler=get_weather)]
    )


@pytest.fixture
def make_agent(library, travel):
    def factory(*script, events=None, **options):
        provider = ScriptedProvider(script)
        agent = PlanningAgent(
            provider,
            library,
            travel,
            today=lambda: TODAY,
            on_event=events.append if events is not None else None,
            **options,
        )
        return agent, provider

    return factory


def test_research_then_submit(make_agent):
    events = []
    agent, provider = make_agent(reply(search()), reply(submit(1)), events=events)
    plan = agent.plan("A day in Examplestad")

    assert plan.itinerary.title == "A day in Examplestad"
    assert [s.number for s in plan.sources] == [1]
    assert plan.source(1).chunk.article == "Examplestad"
    assert plan.source(2) is None
    assert (plan.provider, plan.model) == ("fake", "scripted")
    assert plan.progress.model_calls == 2
    assert plan.progress.usage == Usage(20, 10)
    assert [s.call.name for s in plan.progress.tool_steps] == ["search_guide", "submit_itinerary"]
    assert not plan.used_weather

    assert [type(e) for e in events] == [
        ModelReplied,
        ToolStarted,
        ToolFinished,
        ModelReplied,
        ToolStarted,
        ToolFinished,
    ]
    assert provider.remaining == 0

    first, tools = provider.requests[0]
    assert "Today is Friday 13 November 2026." in first.system
    assert first.turns[0] == UserTurn("<trip_request>\nA day in Examplestad\n</trip_request>")
    assert [t.name for t in tools] == ["search_guide", "get_weather", "submit_itinerary"]
    assert tools[-1] == SUBMIT_SPEC

    second, _ = provider.requests[1]
    (results,) = [t for t in second.turns if isinstance(t, ToolResultsTurn)]
    assert '<passage number="1"' in results.results[0].content


def test_several_tool_calls_in_one_reply(make_agent, weather_calls):
    agent, provider = make_agent(
        reply(search(), call("get_weather", place="Examplestad")),
        reply(submit(1)),
    )
    plan = agent.plan("A day in Examplestad")
    assert weather_calls == ["Examplestad"]
    assert plan.used_weather
    second, _ = provider.requests[1]
    results = second.turns[-1]
    assert isinstance(results, ToolResultsTurn)
    assert [r.call_id for r in results.results] == ["search_guide-1", "get_weather-1"]


def test_invalid_itinerary_is_sent_back_and_can_be_fixed(make_agent):
    broken = itinerary(1)
    broken["days"][0]["day"] = 2
    agent, _ = make_agent(
        reply(search()),
        reply(call("submit_itinerary", 1, **broken)),
        reply(submit(1, n=2)),
    )
    plan = agent.plan("A day in Examplestad")
    first_try = plan.progress.calls_to("submit_itinerary")[0]
    rejected = first_try.result
    assert first_try.call.id == "submit_itinerary-1"
    assert rejected.is_error
    assert "The itinerary has problems" in rejected.content
    assert "numbered 1, 2, 3" in rejected.content
    assert plan.progress.model_calls == 3


def test_invented_citations_are_rejected(make_agent):
    agent, _ = make_agent(reply(search(limit=2)), reply(submit(1, 7, 9)), reply(submit(2, n=2)))
    plan = agent.plan("A day in Examplestad")
    rejected = plan.progress.calls_to("submit_itinerary")[0].result
    assert rejected.is_error
    assert "cites [7], [9], but search_guide never returned those passages" in rejected.content
    assert "known passages: 1 to 2" in rejected.content
    assert [s.number for s in plan.sources] == [2]


def test_tip_citations_are_checked_too(make_agent):
    agent, _ = make_agent(
        reply(search(limit=2)),
        reply(submit(1, tips=["Take the tram [5]."])),
        reply(submit(1, n=2, tips=["Take the tram [2]."])),
    )
    plan = agent.plan("A day in Examplestad")
    assert "cites [5]" in plan.progress.calls_to("submit_itinerary")[0].result.content
    assert [s.number for s in plan.sources] == [1, 2]


def test_submitting_before_any_search_is_rejected(make_agent):
    agent, _ = make_agent(reply(submit(1)), reply(search()), reply(submit(1, n=2)))
    plan = agent.plan("A day in Examplestad")
    rejected = plan.progress.calls_to("submit_itinerary")[0].result
    assert rejected.is_error and "Search the guides with search_guide first" in rejected.content


def test_a_plain_text_reply_gets_one_nudge(make_agent):
    agent, provider = make_agent(
        reply(search()), "Here is your plan: go to the lighthouse.", reply(submit(1))
    )
    plan = agent.plan("A day in Examplestad")
    assert plan.itinerary.days[0].stops[0].name == "Old Lighthouse"
    last, _ = provider.requests[-1]
    assert isinstance(last.turns[-1], UserTurn)
    assert "call submit_itinerary" in last.turns[-1].text


def test_giving_up_after_the_nudge(make_agent):
    agent, _ = make_agent(reply(search()), "Sorry.", "I can't plan this trip.")
    with pytest.raises(PlanningError, match="stopped without submitting") as info:
        agent.plan("A trip to the Moon")
    assert info.value.model_text == "I can't plan this trip."
    assert info.value.progress.model_calls == 3
    assert len(info.value.progress.tool_steps) == 1


def test_refusal(make_agent):
    refusal = LLMResponse(
        text="I can't help with that.",
        stop_reason=StopReason.REFUSAL,
        provider="fake",
        model="scripted",
    )
    agent, _ = make_agent(refusal)
    with pytest.raises(PlanningError, match="declined"):
        agent.plan("Something harmful")


def test_running_out_of_tokens(make_agent):
    cut_off = LLMResponse(
        text="Day 1: ...", stop_reason=StopReason.MAX_TOKENS, provider="fake", model="scripted"
    )
    agent, _ = make_agent(reply(search()), cut_off)
    with pytest.raises(PlanningError, match="ran out of room"):
        agent.plan("21 days everywhere")


def test_step_limit(make_agent):
    agent, provider = make_agent(
        *[reply(search(f"query {n}", n=n)) for n in range(1, 5)], max_steps=3
    )
    with pytest.raises(PlanningError, match="No itinerary after 3 steps") as info:
        agent.plan("A day in Examplestad")
    assert len(info.value.progress.tool_steps) == 3
    assert provider.remaining == 1


def test_tool_errors_go_back_to_the_model(make_agent):
    agent, _ = make_agent(
        reply(call("teleport", to="Mars"), call("search_guide", destination="Nowhere", query="x")),
        reply(search()),
        reply(submit(1)),
    )
    plan = agent.plan("A day in Examplestad")
    first_results = [s.result for s in plan.progress.tool_steps[:2]]
    assert all(r.is_error for r in first_results)
    assert "Unknown tool 'teleport'" in first_results[0].content
    assert plan.progress.model_calls == 3


def test_provider_errors_propagate(make_agent):
    agent, _ = make_agent(reply(search()))  # nothing scripted for the second call
    with pytest.raises(LLMError):
        agent.plan("A day in Examplestad")


def test_each_plan_starts_with_fresh_sources(make_agent):
    agent, _ = make_agent(
        reply(search("tram")), reply(submit(1)), reply(search("fish soup")), reply(submit(1))
    )
    first = agent.plan("Trams")
    second = agent.plan("Soup")
    assert first.sources[0].chunk.id != second.sources[0].chunk.id


@pytest.mark.parametrize("request_text", ["", "   "])
def test_empty_request(make_agent, request_text):
    agent, _ = make_agent()
    with pytest.raises(ValueError, match="empty"):
        agent.plan(request_text)


def test_option_checks(library):
    with pytest.raises(ValueError, match="max_steps"):
        PlanningAgent(ScriptedProvider([]), library, max_steps=1)

    class Args(BaseModel):
        pass

    clash = ToolRegistry(
        [Tool(name="submit_itinerary", description="d", args_model=Args, handler=lambda a: 1)]
    )
    with pytest.raises(ValueError, match="reserved"):
        PlanningAgent(ScriptedProvider([]), library, clash).plan("A trip")


def test_plan_sources_are_listed_by_number(make_agent):
    agent, _ = make_agent(reply(search(limit=3)), reply(submit(3, 1)))
    plan = agent.plan("A day in Examplestad")
    assert [s.number for s in plan.sources] == [1, 3]
