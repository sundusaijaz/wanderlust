import json

import anthropic
import httpx2
import pytest

from wanderlust.llm.anthropic_provider import (
    DEFAULT_MODEL,
    FALLBACK_BETA,
    AnthropicProvider,
    echo_blocks,
)
from wanderlust.llm.fake import ScriptedProvider
from wanderlust.llm.types import (
    Conversation,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
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


def message(content, stop_reason="end_turn", model=DEFAULT_MODEL):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


def make_provider(handler, **kwargs):
    requests: list[httpx2.Request] = []

    def recording(request):
        requests.append(request)
        return handler(request)

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(recording))
    provider = AnthropicProvider("test-key", max_retries=0, http_client=http, **kwargs)
    return provider, requests


def body(request) -> dict:
    return json.loads(request.content)


def test_satisfies_protocol():
    provider, _ = make_provider(lambda r: httpx2.Response(200, json=message([])))
    assert isinstance(provider, LLMProvider)
    assert provider.model == "claude-opus-5-5"


def test_complete_sends_expected_request_and_parses_text():
    provider, requests = make_provider(
        lambda r: httpx2.Response(200, json=message([{"type": "text", "text": "Hello!"}]))
    )
    response = provider.complete(Conversation(system="Be brief.").add_user("Hi"))

    assert response.text == "Hello!"
    assert response.stop_reason is StopReason.END
    assert response.provider == "anthropic"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 7)

    sent = body(requests[0])
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"] == "Be brief."
    assert sent["messages"] == [{"role": "user", "content": "Hi"}]
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["fallbacks"] == "default"
    assert "thinking" not in sent and "temperature" not in sent
    assert FALLBACK_BETA in requests[0].headers["anthropic-beta"]
    assert requests[0].headers["authorization"] == "Bearer test-key"


def test_fallbacks_are_skipped_for_models_without_them():
    provider, requests = make_provider(
        lambda r: httpx2.Response(200, json=message([], model="claude-haiku-5-5")),
        model="claude-haiku-5-5",
    )
    provider.complete(Conversation().add_user("Hi"))
    assert "fallbacks" not in body(requests[0])
    assert "anthropic-beta" not in requests[0].headers


def test_tool_use_round_trip_replays_assistant_blocks_unchanged():
    replies = iter(
        [
            message(
                [
                    {"type": "thinking", "thinking": "", "signature": "sig-abc"},
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Lisbon"},
                    },
                ],
                stop_reason="tool_use",
            ),
            message([{"type": "text", "text": "Sunny, 21°C."}]),
        ]
    )
    provider, requests = make_provider(lambda r: httpx2.Response(200, json=next(replies)))

    convo = Conversation().add_user("Weather in Lisbon?")
    first = provider.complete(convo, tools=[WEATHER])
    assert first.stop_reason is StopReason.TOOL_USE
    assert first.tool_calls == [
        ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Lisbon"})
    ]

    convo.add_response(first).add_tool_results(
        [ToolResult(call_id="toolu_1", content="Sunny, 21°C")]
    )
    second = provider.complete(convo, tools=[WEATHER])
    assert second.text == "Sunny, 21°C."

    sent_tools = body(requests[0])["tools"]
    assert sent_tools == [
        {
            "name": "get_weather",
            "description": "Forecast for a city",
            "input_schema": WEATHER.parameters,
        }
    ]
    messages = body(requests[1])["messages"]
    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    assert assistant["content"][2]["type"] == "tool_use"
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "Sunny, 21°C",
                "is_error": False,
            }
        ],
    }


def test_turns_from_other_providers_are_rebuilt():
    provider, requests = make_provider(
        lambda r: httpx2.Response(200, json=message([{"type": "text", "text": "ok"}]))
    )
    foreign = ScriptedProvider.tool_request(
        ToolCall(id="c1", name="get_weather", arguments={"city": "Porto"}), text="Checking"
    )
    convo = (
        Conversation()
        .add_user("Weather?")
        .add_response(foreign)
        .add_tool_results([ToolResult(call_id="c1", content="Rain", is_error=False)])
    )
    provider.complete(convo)
    assert body(requests[0])["messages"][1]["content"] == [
        {"type": "text", "text": "Checking"},
        {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Porto"}},
    ]


def test_echo_blocks_drops_declined_attempt_before_fallback():
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "s1"},
        {"type": "text", "text": "Partial "},
        {"type": "tool_use", "id": "t0", "name": "x", "input": {}},
        {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
        {"type": "thinking", "thinking": "", "signature": "s2"},
        {"type": "text", "text": "answer"},
    ]
    assert echo_blocks(blocks) == [
        {"type": "text", "text": "Partial "},
        {"type": "thinking", "thinking": "", "signature": "s2"},
        {"type": "text", "text": "answer"},
    ]
    plain = [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "x"}]
    assert echo_blocks(plain) == plain


@pytest.mark.parametrize(
    ("anthropic_reason", "expected"),
    [
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
        ("pause_turn", StopReason.OTHER),
    ],
)
def test_stop_reasons_are_normalised(anthropic_reason, expected):
    provider, _ = make_provider(
        lambda r: httpx2.Response(200, json=message([], stop_reason=anthropic_reason))
    )
    assert provider.complete(Conversation().add_user("x")).stop_reason is expected


def sse(*events) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def test_stream_yields_text_deltas():
    start = message([], stop_reason=None)
    stream_body = sse(
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Obrigado"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " and welcome!"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    )
    provider, requests = make_provider(
        lambda r: httpx2.Response(
            200, content=stream_body, headers={"content-type": "text/event-stream"}
        )
    )
    pieces = list(provider.stream(Conversation().add_user("Say thanks in Portuguese")))
    assert pieces == ["Obrigado", " and welcome!"]
    assert body(requests[0])["stream"] is True


def error_response(status, error_type):
    return lambda r: httpx2.Response(
        status, json={"type": "error", "error": {"type": error_type, "message": "nope"}}
    )


@pytest.mark.parametrize(
    ("status", "error_type", "expected"),
    [
        (401, "authentication_error", LLMAuthError),
        (403, "permission_error", LLMAuthError),
        (429, "rate_limit_error", LLMRateLimitError),
        (529, "overloaded_error", LLMRateLimitError),
        (400, "invalid_request_error", LLMError),
    ],
)
def test_api_errors_are_translated(status, error_type, expected):
    provider, _ = make_provider(error_response(status, error_type))
    with pytest.raises(expected):
        provider.complete(Conversation().add_user("x"))


def test_connection_errors_are_translated():
    def offline(request):
        raise httpx2.ConnectError("offline", request=request)

    provider, _ = make_provider(offline)
    with pytest.raises(LLMConnectionError):
        provider.complete(Conversation().add_user("x"))
    with pytest.raises(LLMConnectionError):
        list(provider.stream(Conversation().add_user("x")))
