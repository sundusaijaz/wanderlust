import json

import pytest

from wanderlust.agent.planner import Progress
from wanderlust.llm.types import ToolCall, ToolResult, Usage
from wanderlust.plan_view import describe_call, progress_summary, summarize_result


def call(name, **arguments):
    return ToolCall(id="c", name=name, arguments=arguments)


def ok(content):
    return ToolResult(
        call_id="c", content=content if isinstance(content, str) else json.dumps(content)
    )


@pytest.mark.parametrize(
    ("tool_call", "text"),
    [
        (call("search_guide", destination="Lisbon", query="tram 28"), "Searching the Lisbon guide"),
        (call("geocode", place="Sintra"), "Looking up Sintra"),
        (call("get_weather", start_date="2027-03-10"), "weather from 2027-03-10"),
        (call("get_distance"), "Estimating travel times"),
        (call("submit_itinerary"), "Checking the itinerary"),
        (call("teleport"), "Running teleport"),
    ],
)
def test_describe_call(tool_call, text):
    assert text in describe_call(tool_call)


def test_summaries():
    assert summarize_result(call("search_guide"), ok('<passage n="1">x</passage>')) == "1 passage"
    assert summarize_result(call("search_guide"), ok("No passages match.")) == "nothing found"
    assert summarize_result(call("submit_itinerary"), ok("Itinerary accepted.")) == "accepted"
    typical = {"kind": "typical", "days": [{"plan_indoor_alternatives": True}, {}]}
    assert summarize_result(call("get_weather"), ok(typical)) == "typical weather, 2 days, 1 wet"
    assert summarize_result(call("get_distance"), ok({"straight_line_km": 21.46})) == "21.5 km"
    assert summarize_result(call("other"), ok("done\nmore")) == "done"


def test_error_summaries_are_one_short_line():
    error = ToolResult(call_id="c", content="x" * 300 + "\nsecond line", is_error=True)
    summary = summarize_result(call("geocode"), error)
    assert len(summary) == 160 and summary.endswith("...")


def test_progress_summary():
    progress = Progress(model_calls=1, usage=Usage(1200, 300))
    assert progress_summary(progress) == "1 model call, 0 tool calls, 1,500 tokens"
    assert progress_summary(Progress()) == "0 model calls, 0 tool calls"
