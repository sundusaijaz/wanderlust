import json

import httpx
import pytest
from pydantic import BaseModel

from wanderlust.llm.types import ToolCall
from wanderlust.tools.http import ToolError
from wanderlust.tools.registry import (
    Tool,
    ToolRegistry,
    build_tools_client,
    json_schema,
    travel_tools,
)

from .test_geocoding import geocoding_api
from .test_weather import archive_api, forecast_api


def all_apis(request):
    host = request.url.host
    if host.startswith("geocoding"):
        return geocoding_api(request)
    if host.startswith("archive"):
        return archive_api(request)
    return forecast_api(request)


@pytest.fixture
def tools(make_meteo):
    client, requests = make_meteo(all_apis)
    return travel_tools(client), requests


def call(name, **arguments):
    return ToolCall(id=f"call-{name}", name=name, arguments=arguments)


def test_specs_describe_every_tool(tools):
    registry, _ = tools
    specs = {spec.name: spec for spec in registry.specs}
    assert set(specs) == {"geocode", "get_weather", "get_distance"}

    weather = specs["get_weather"].parameters
    assert weather["type"] == "object"
    assert set(weather["required"]) == {"latitude", "longitude", "start_date", "end_date"}
    assert weather["properties"]["start_date"]["format"] == "date"
    assert weather["additionalProperties"] is False
    assert "title" not in json.dumps(weather)
    assert "2026-11-27" in specs["get_weather"].description  # forecast horizon


def test_geocode_tool(tools):
    registry, _ = tools
    result = registry.run(call("geocode", place="Examplestad", country_code="IM"))
    assert not result.is_error
    assert result.call_id == "call-geocode"
    (match,) = json.loads(result.content)["matches"]
    assert match["label"] == "Examplestad, Coast Province, Imaginaria"


def test_geocode_tool_not_found(tools):
    registry, _ = tools
    result = registry.run(call("geocode", place="Nowhereville"))
    assert result.is_error and "No place called 'Nowhereville'" in result.content


def test_weather_tool_forecast_and_typical(tools):
    registry, _ = tools
    soon = json.loads(
        registry.run(
            call(
                "get_weather",
                latitude=1,
                longitude=2,
                start_date="2026-11-13",
                end_date="2026-11-14",
            )
        ).content
    )
    assert soon["kind"] == "forecast"
    assert soon["days"][1]["plan_indoor_alternatives"] is True
    assert soon["days"][0]["summary"] == "Clear sky"
    assert "Open-Meteo" in soon["attribution"]

    later = json.loads(
        registry.run(
            call(
                "get_weather",
                latitude=1,
                longitude=2,
                start_date="2027-03-01",
                end_date="2027-03-02",
            )
        ).content
    )
    assert later["kind"] == "typical"
    assert "not a forecast" in later["note"]


def test_distance_tool(tools):
    registry, requests = tools
    result = registry.run(
        call(
            "get_distance",
            from_latitude=38.7139,
            from_longitude=-9.1394,
            to_latitude=38.6916,
            to_longitude=-9.2160,
        )
    )
    payload = json.loads(result.content)
    assert 6 < payload["straight_line_km"] < 8
    assert {e["mode"] for e in payload["estimates"]} >= {"transit", "drive"}
    assert requests == []  # no network needed


@pytest.mark.parametrize(
    ("tool_call", "message"),
    [
        (call("teleport", to="Mars"), "Unknown tool 'teleport'"),
        (call("geocode"), "place: Field required"),
        (call("geocode", place="Lisbon", extra=1), "extra: Extra inputs are not permitted"),
        (
            call(
                "get_weather",
                latitude=200,
                longitude=0,
                start_date="2026-11-13",
                end_date="2026-11-13",
            ),
            "latitude",
        ),
        (
            call(
                "get_weather",
                latitude=1,
                longitude=2,
                start_date="2026-11-01",
                end_date="2026-11-02",
            ),
            "past",
        ),
    ],
)
def test_bad_calls_become_error_results(tools, tool_call, message):
    registry, _ = tools
    result = registry.run(tool_call)
    assert result.is_error
    assert message in result.content


def test_tool_errors_are_reported_not_raised():
    class Args(BaseModel):
        x: int

    def boom(args):
        raise ToolError("service down")

    registry = ToolRegistry([Tool(name="boom", description="d", args_model=Args, handler=boom)])
    result = registry.run(call("boom", x=1))
    assert result.is_error and result.content == "service down"


def test_json_schema_strips_titles():
    class Args(BaseModel):
        city: str

    assert json_schema(Args) == {
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "type": "object",
    }


def test_build_tools_client_caches_under_data_dir(tmp_path):
    client = build_tools_client(tmp_path)
    try:
        assert client.cache.root == tmp_path / "http-cache" / "open-meteo"
    finally:
        client.close()


def test_unexpected_exceptions_still_propagate():
    class Args(BaseModel):
        x: int

    def bug(args):
        raise KeyError("programming error")

    registry = ToolRegistry([Tool(name="bug", description="d", args_model=Args, handler=bug)])
    with pytest.raises(KeyError):
        registry.run(call("bug", x=1))


def test_forecast_api_shape_is_used(make_meteo):
    # Guard against accidentally calling the archive for near dates.
    seen = []

    def api(request):
        seen.append(request.url.host)
        return all_apis(request)

    client, _ = make_meteo(api)
    travel_tools(client).run(
        call("get_weather", latitude=1, longitude=2, start_date="2026-11-13", end_date="2026-11-13")
    )
    assert seen == ["api.open-meteo.com"]


def test_http_errors_from_open_meteo_become_tool_errors(make_meteo):
    client, _ = make_meteo(lambda r: httpx.Response(500, json={}))
    result = travel_tools(client).run(call("geocode", place="Lisbon"))
    assert result.is_error and "HTTP 500" in result.content


def test_json_schema_inlines_nested_models():
    class Stop(BaseModel):
        name: str

    class Day(BaseModel):
        stops: list[Stop]
        highlight: Stop | None = None

    schema = json_schema(Day)
    assert "$defs" not in json.dumps(schema)
    assert schema["properties"]["stops"]["items"] == {
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "type": "object",
    }
    options = schema["properties"]["highlight"]["anyOf"]
    assert {"type": "null"} in options
    assert any(option.get("required") == ["name"] for option in options)


def test_json_schema_rejects_recursive_models():
    class Node(BaseModel):
        children: list["Node"] = []

    with pytest.raises(ValueError, match="recursive"):
        json_schema(Node)


def _tool(name):
    class Args(BaseModel):
        pass

    return Tool(name=name, description=name, args_model=Args, handler=lambda args: name)


def test_registries_can_be_combined():
    combined = ToolRegistry([_tool("a")]) + ToolRegistry([_tool("b")])
    assert combined.names == ["a", "b"]
    assert "b" in combined and "c" not in combined
    assert combined.run(call("b")).content == "b"


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate tool name 'a'"):
        ToolRegistry([_tool("a")]) + ToolRegistry([_tool("a")])
