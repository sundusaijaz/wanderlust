import json

import httpx2
import openai
import pytest

from wanderlust.llm.anthropic_provider import AnthropicProvider
from wanderlust.llm.openai_provider import OpenAIProvider
from wanderlust.llm.types import (
    Conversation,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
)

WEATHER = ToolSpec(
    name="get_weather",
    description="Forecast for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def completion(message, finish_reason="stop"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5-mini",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def make_provider(handler, **kwargs):
    requests: list[httpx2.Request] = []

    def recording(request):
        requests.append(request)
        return handler(request)

    http = openai.DefaultHttpxClient(transport=httpx2.MockTransport(recording))
    return OpenAIProvider("test-key", max_retries=0, http_client=http, **kwargs), requests


def body(request):
    return json.loads(request.content)


def test_satisfies_protocol():
    provider, _ = make_provider(lambda r: httpx2.Response(500))
    assert isinstance(provider, LLMProvider)
    assert provider.model == "gpt-5-mini"


def test_complete_sends_system_and_parses_text():
    provider, requests = make_provider(
        lambda r: httpx2.Response(200, json=completion({"role": "assistant", "content": "Olá!"}))
    )
    response = provider.complete(Conversation(system="Be brief.").add_user("Hi"), max_tokens=500)
    assert response.text == "Olá!"
    assert response.stop_reason is StopReason.END
    assert (response.usage.input_tokens, response.usage.output_tokens) == (20, 5)

    sent = body(requests[0])
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    assert sent["max_completion_tokens"] == 500
    assert "reasoning_effort" not in sent
    assert requests[0].headers["authorization"] == "Bearer test-key"


def test_reasoning_effort_is_passed_when_set():
    provider, requests = make_provider(
        lambda r: httpx2.Response(200, json=completion({"role": "assistant", "content": "ok"})),
        reasoning_effort="low",
    )
    provider.complete(Conversation().add_user("x"))
    assert body(requests[0])["reasoning_effort"] == "low"


def test_tool_call_round_trip():
    replies = iter(
        [
            completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'},
                        }
                    ],
                },
                finish_reason="tool_calls",
            ),
            completion({"role": "assistant", "content": "Sunny."}),
        ]
    )
    provider, requests = make_provider(lambda r: httpx2.Response(200, json=next(replies)))

    convo = Conversation().add_user("Weather?")
    first = provider.complete(convo, tools=[WEATHER])
    assert first.stop_reason is StopReason.TOOL_USE
    assert first.tool_calls == [
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Lisbon"})
    ]

    convo.add_response(first).add_tool_results(
        [ToolResult(call_id="call_1", content="no data", is_error=True)]
    )
    assert provider.complete(convo, tools=[WEATHER]).text == "Sunny."

    assert body(requests[0])["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Forecast for a city",
                "parameters": WEATHER.parameters,
            },
        }
    ]
    messages = body(requests[1])["messages"]
    assert messages[1]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Lisbon"}',
    }
    assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "Error: no data"}


def test_turns_from_anthropic_are_translated():
    provider, requests = make_provider(
        lambda r: httpx2.Response(200, json=completion({"role": "assistant", "content": "ok"}))
    )
    claude_turn = LLMResponse(
        text="Checking",
        stop_reason=StopReason.TOOL_USE,
        provider=AnthropicProvider.name,
        model="claude-opus-5-5",
        tool_calls=[ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Porto"})],
        raw=[{"type": "thinking", "thinking": "", "signature": "s"}],
    )
    convo = (
        Conversation()
        .add_user("?")
        .add_response(claude_turn)
        .add_tool_results([ToolResult(call_id="toolu_1", content="Rain")])
    )
    provider.complete(convo)
    assistant = body(requests[0])["messages"][1]
    assert assistant["content"] == "Checking"
    assert assistant["tool_calls"][0]["id"] == "toolu_1"


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("length", StopReason.MAX_TOKENS), ("content_filter", StopReason.REFUSAL)],
)
def test_finish_reasons_are_normalised(finish_reason, expected):
    provider, _ = make_provider(
        lambda r: httpx2.Response(
            200, json=completion({"role": "assistant", "content": "x"}, finish_reason)
        )
    )
    assert provider.complete(Conversation().add_user("x")).stop_reason is expected


def test_refusal_message_becomes_refusal_stop():
    provider, _ = make_provider(
        lambda r: httpx2.Response(
            200,
            json=completion({"role": "assistant", "content": None, "refusal": "I can't help."}),
        )
    )
    response = provider.complete(Conversation().add_user("x"))
    assert response.stop_reason is StopReason.REFUSAL
    assert response.text == "I can't help."


def test_malformed_tool_arguments_raise():
    provider, _ = make_provider(
        lambda r: httpx2.Response(
            200,
            json=completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{oops"},
                        }
                    ],
                },
                "tool_calls",
            ),
        )
    )
    with pytest.raises(LLMError, match="malformed arguments"):
        provider.complete(Conversation().add_user("x"))


def test_stream_yields_content_deltas():
    def chunk(content):
        delta = {"content": content} if content is not None else {}
        return {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }

    stream_body = (
        "".join(f"data: {json.dumps(chunk(c))}\n\n" for c in ["Bom", " dia", None])
        + "data: [DONE]\n\n"
    )
    provider, requests = make_provider(
        lambda r: httpx2.Response(
            200, content=stream_body.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    assert list(provider.stream(Conversation().add_user("Greet me"))) == ["Bom", " dia"]
    assert body(requests[0])["stream"] is True


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, LLMAuthError), (403, LLMAuthError), (429, LLMRateLimitError), (400, LLMError)],
)
def test_api_errors_are_translated(status, expected):
    provider, _ = make_provider(
        lambda r: httpx2.Response(status, json={"error": {"message": "nope", "type": "x"}})
    )
    with pytest.raises(expected):
        provider.complete(Conversation().add_user("x"))


def test_connection_errors_are_translated():
    def offline(request):
        raise httpx2.ConnectError("offline", request=request)

    provider, _ = make_provider(offline)
    with pytest.raises(LLMConnectionError):
        provider.complete(Conversation().add_user("x"))
