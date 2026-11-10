import pytest

from wanderlust.llm.fake import ScriptedProvider
from wanderlust.llm.types import (
    Conversation,
    LLMError,
    LLMProvider,
    StopReason,
    ToolCall,
    ToolSpec,
)


def test_fake_satisfies_the_protocol():
    assert isinstance(ScriptedProvider([]), LLMProvider)


def test_replies_in_order_and_records_requests():
    fake = ScriptedProvider(["First answer.", "Second answer."])
    convo = Conversation(system="sys").add_user("hello there")
    tool = ToolSpec(name="weather", description="Get weather", parameters={"type": "object"})

    first = fake.complete(convo, tools=[tool])
    assert first.text == "First answer."
    assert first.stop_reason is StopReason.END
    assert first.usage.input_tokens == 2

    convo.add_response(first).add_user("again")
    assert fake.complete(convo).text == "Second answer."

    assert fake.remaining == 0
    seen_convo, seen_tools = fake.requests[0]
    assert len(seen_convo.turns) == 1  # a snapshot, not the live object
    assert seen_tools == [tool]


def test_scripted_tool_requests():
    call = ToolCall(id="c1", name="weather", arguments={"city": "Porto"})
    fake = ScriptedProvider([ScriptedProvider.tool_request(call)])
    response = fake.complete(Conversation().add_user("?"))
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.tool_calls == [call]


def test_stream_yields_pieces_that_join_back():
    fake = ScriptedProvider(["Lisbon is  hilly.\nBring good shoes."])
    pieces = list(fake.stream(Conversation().add_user("tips?")))
    assert len(pieces) > 1
    assert "".join(pieces) == "Lisbon is  hilly.\nBring good shoes."


def test_running_out_of_script_is_an_error():
    with pytest.raises(LLMError, match="no responses left"):
        ScriptedProvider([]).complete(Conversation().add_user("?"))
