import pytest

from wanderlust.llm.types import (
    AssistantTurn,
    Conversation,
    LLMResponse,
    StopReason,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    UserTurn,
)


def test_conversation_is_built_in_order():
    response = LLMResponse(
        text="",
        stop_reason=StopReason.TOOL_USE,
        provider="fake",
        model="m",
        tool_calls=[ToolCall(id="1", name="weather", arguments={"city": "Lisbon"})],
    )
    convo = (
        Conversation(system="Be brief.")
        .add_user("Weather in Lisbon?")
        .add_response(response)
        .add_tool_results([ToolResult(call_id="1", content="Sunny, 21°C")])
    )
    assert [type(t) for t in convo.turns] == [UserTurn, AssistantTurn, ToolResultsTurn]
    assert convo.turns[1].response.tool_calls[0].arguments == {"city": "Lisbon"}


def test_tool_results_cannot_be_empty():
    with pytest.raises(ValueError):
        Conversation().add_tool_results([])


def test_raw_payload_is_ignored_in_equality():
    a = LLMResponse(text="hi", stop_reason=StopReason.END, provider="p", model="m", raw={"x": 1})
    b = LLMResponse(text="hi", stop_reason=StopReason.END, provider="p", model="m", raw={"x": 2})
    assert a == b
