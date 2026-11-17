import logging
import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from wanderlust.config import Settings
from wanderlust.llm.factory import build_provider
from wanderlust.llm.fake import ScriptedProvider
from wanderlust.llm.types import ToolCall
from wanderlust.web.app import PlanRequest, _plan_events, create_app, sse

from .conftest import parse_events


def call(name, n=1, **arguments):
    return ToolCall(id=f"{name}-{n}", name=name, arguments=arguments)


ITINERARY = {
    "title": "A day in Examplestad",
    "summary": "Lighthouse at sunset.",
    "days": [
        {
            "day": 1,
            "base": "Examplestad",
            "title": "The lighthouse",
            "stops": [
                {
                    "time_of_day": "evening",
                    "name": "Old Lighthouse",
                    "description": "Climb the 120 steps at sunset.",
                    "sources": [1],
                }
            ],
        }
    ],
}


def happy_script():
    return [
        ScriptedProvider.tool_request(
            call("search_guide", destination="Examplestad", query="lighthouse"),
            call("geocode", place="Examplestad"),
            text="Researching the guide.",
        ),
        ScriptedProvider.tool_request(call("submit_itinerary", **ITINERARY)),
    ]


def plan(web, request="A day in Examplestad", **extra):
    response = web.post("/api/plan", json={"request": request, **extra})
    return response, parse_events(response.text) if response.status_code == 200 else []


def test_status_reports_the_model(web):
    status = web.get("/api/status").json()
    assert status["ready"] is True
    assert status["provider"] == "fake" and status["model"] == "scripted"
    assert status["problem"] is None and status["demo"] is False
    assert status["version"]


def test_index_and_static_files(web):
    page = web.get("/")
    assert page.status_code == 200 and "Wanderlust" in page.text
    for asset in ("/static/app.js", "/static/app.css", "/static/favicon.svg"):
        assert asset in page.text
        assert web.get(asset).status_code == 200
    assert web.get("/static/../app.py").status_code == 404


def test_page_never_renders_model_text_as_html(web):
    script = web.get("/static/app.js").text
    assert ".innerHTML" not in script and "insertAdjacentHTML" not in script


def test_plan_streams_progress_then_the_itinerary(web, script):
    script.extend(happy_script())
    response, events = plan(web)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in events]
    assert names == [
        "start",
        "thinking",
        "tool_started",
        "status",  # the guide index is built during the first search
        "tool_finished",
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
        "plan",
        "done",
    ]
    data = dict(events[:4])
    assert data["start"] == {"provider": "fake", "model": "scripted"}
    assert "Building the Examplestad guide index" in data["status"]["message"]
    assert data["thinking"]["text"] == "Researching the guide."

    started = events[2][1]
    assert started == {
        "id": "search_guide-1",
        "name": "search_guide",
        "label": "Searching the Examplestad guide for 'lighthouse'",
    }
    finished = [d for name, d in events if name == "tool_finished"]
    assert finished[0]["summary"] == "5 passages" and not finished[0]["is_error"]
    assert finished[1]["summary"] == "Examplestad, Imaginaria"
    assert finished[2]["summary"] == "accepted"

    result = events[-2][1]
    assert result["itinerary"]["days"][0]["stops"][0]["name"] == "Old Lighthouse"
    assert [s["number"] for s in result["sources"]] == [1]
    assert result["sources"][0]["url"].startswith("https://en.wikivoyage.org/wiki/Examplestad")
    assert result["summary"] == "2 model calls, 3 tool calls"
    assert result["used_weather"] is False
    assert result["request"] == "A day in Examplestad"


def test_planning_failure_is_an_error_event(web, script):
    script.extend(["I need more details.", "Sorry, I can't plan that."])
    _, events = plan(web, "Somewhere")
    assert [name for name, _ in events][-2:] == ["error", "done"]
    error = events[-2][1]
    assert "stopped without submitting" in error["message"]
    assert error["model_text"] == "Sorry, I can't plan that."
    assert error["summary"].startswith("2 model calls, 0 tool calls")


def test_provider_errors_are_error_events(web):
    _, events = plan(web)  # the script is empty: the model call fails
    assert events[-2] == ("error", {"message": "ScriptedProvider has no responses left"})


def test_unexpected_errors_are_reported_without_details(settings, services, caplog):
    def broken(settings):
        raise RuntimeError("secret internals")

    web = TestClient(create_app(settings, replace(services, tools_client=broken)))
    _, events = plan(web)
    message = events[-2][1]["message"]
    assert message == "Something went wrong while planning. Please try again."
    assert "secret internals" not in message
    assert "Planning failed" in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        {"request": "hi"},
        {"request": "x" * 501},
        {"request": "A day in Examplestad", "max_steps": 99},
        {"request": "A day in Examplestad", "admin": True},
        {},
    ],
)
def test_invalid_requests_are_rejected(web, body):
    assert web.post("/api/plan", json=body).status_code == 422


def test_max_steps_is_passed_to_the_agent(web, script):
    script.extend(happy_script())
    _, events = plan(web, max_steps=2)
    assert events[-2][0] == "plan"  # two model calls are just enough


def test_busy_planner_returns_429(settings, services):
    release = threading.Event()

    def slow_provider(settings):
        release.wait(5)
        return ScriptedProvider([])

    app = create_app(settings, replace(services, provider=slow_provider), max_concurrent_plans=1)
    web = TestClient(app)
    results = []
    first = threading.Thread(target=lambda: results.append(plan(web)[0].status_code))
    first.start()
    time.sleep(0.2)
    try:
        assert web.post("/api/plan", json={"request": "Another trip"}).status_code == 429
    finally:
        release.set()
        first.join(5)
    assert results == [200]
    assert web.post("/api/plan", json={"request": "Third trip"}).status_code == 200


def test_keepalive_comments_while_waiting(settings, services, script):
    def slow_provider(settings):
        time.sleep(0.3)
        return ScriptedProvider(script)

    app = create_app(settings, replace(services, provider=slow_provider), keepalive_seconds=0.05)
    response = TestClient(app).post("/api/plan", json={"request": "A day in Examplestad"})
    assert ": keep-alive" in response.text
    assert parse_events(response.text)[-1][0] == "done"


def test_sse_format():
    assert sse("status", {"message": "Hi\nthere"}) == (
        'event: status\ndata: {"message": "Hi\\nthere"}\n\n'
    )


def test_status_problem_without_key(services, tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    app = create_app(settings, replace(services, provider=build_provider), demo=True)
    status = TestClient(app).get("/api/status").json()
    assert status["ready"] is False
    assert "OPENAI_API_KEY" in status["problem"]
    assert status["provider"] == "openai" and status["model"] is None
    assert status["demo"] is True


def test_planning_stops_when_the_client_disconnects(settings, services, caplog):
    gate = threading.Event()
    provider = ScriptedProvider(happy_script())
    original = provider.complete

    def waiting_complete(*args, **kwargs):
        gate.wait(5)
        return original(*args, **kwargs)

    provider.complete = waiting_complete
    caplog.set_level(logging.INFO, logger="wanderlust.web.app")
    events = _plan_events(
        PlanRequest(request="A day in Examplestad"),
        settings,
        replace(services, provider=lambda s: provider),
        keepalive_seconds=1,
    )
    assert next(events).startswith("event: start")
    events.close()  # the browser went away
    gate.set()

    worker = next(t for t in threading.enumerate() if t.name == "wanderlust-plan")
    worker.join(5)
    assert not worker.is_alive()
    assert "Planning cancelled" in caplog.text
    assert provider.remaining == 1  # the model was not asked again
