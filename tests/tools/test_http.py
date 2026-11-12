from datetime import timedelta

import httpx
import pytest

from wanderlust.tools.http import USER_AGENT, JsonCache, OpenMeteoClient, ToolError

URL = "https://api.open-meteo.com/v1/forecast"


def ok(request):
    return httpx.Response(200, json={"echo": dict(request.url.params)})


def test_responses_are_cached_until_they_expire(make_meteo, clock):
    client, requests = make_meteo(ok)
    params = {"latitude": 1, "longitude": 2}

    first = client.get_json(URL, params, ttl=timedelta(hours=1))
    second = client.get_json(URL, params, ttl=timedelta(hours=1))
    assert first == second == {"echo": {"latitude": "1", "longitude": "2"}}
    assert len(requests) == 1

    clock.advance(hours=2)
    client.get_json(URL, params, ttl=timedelta(hours=1))
    assert len(requests) == 2


def test_different_params_are_cached_separately(make_meteo):
    client, requests = make_meteo(ok)
    client.get_json(URL, {"latitude": 1}, ttl=timedelta(hours=1))
    client.get_json(URL, {"latitude": 2}, ttl=timedelta(hours=1))
    assert len(requests) == 2


def test_cache_key_ignores_param_order():
    assert JsonCache.key(URL, {"a": 1, "b": 2}) == JsonCache.key(URL, {"b": 2, "a": 1})


def test_corrupt_cache_entry_is_refetched(make_meteo, tmp_path):
    client, requests = make_meteo(ok)
    client.get_json(URL, {"x": 1}, ttl=timedelta(days=1))
    for path in (tmp_path / "http").rglob("*.json"):
        path.write_text("{broken")
    client.get_json(URL, {"x": 1}, ttl=timedelta(days=1))
    assert len(requests) == 2


def test_open_meteo_error_reason_is_reported(make_meteo):
    client, _ = make_meteo(
        lambda r: httpx.Response(400, json={"error": True, "reason": "start_date out of range"})
    )
    with pytest.raises(ToolError, match="start_date out of range"):
        client.get_json(URL, {}, ttl=timedelta(hours=1))


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(503, json={}), "HTTP 503"),
        (httpx.Response(200, text="<html>"), "invalid JSON"),
        (httpx.Response(200, json=[1, 2]), "unexpected response"),
    ],
)
def test_bad_responses(make_meteo, response, message):
    client, _ = make_meteo(lambda r: response)
    with pytest.raises(ToolError, match=message):
        client.get_json(URL, {}, ttl=timedelta(hours=1))


def test_errors_are_not_cached(make_meteo):
    responses = iter([httpx.Response(503, json={}), httpx.Response(200, json={"fine": True})])
    client, _ = make_meteo(lambda r: next(responses))
    with pytest.raises(ToolError):
        client.get_json(URL, {}, ttl=timedelta(hours=1))
    assert client.get_json(URL, {}, ttl=timedelta(hours=1)) == {"fine": True}


def test_network_failure(make_meteo):
    def offline(request):
        raise httpx.ConnectError("offline", request=request)

    client, _ = make_meteo(offline)
    with pytest.raises(ToolError, match="Could not reach Open-Meteo"):
        client.get_json(URL, {}, ttl=timedelta(hours=1))


def test_default_client_identifies_itself():
    client = OpenMeteoClient()
    try:
        assert client._http.headers["User-Agent"] == USER_AGENT
    finally:
        client.close()
