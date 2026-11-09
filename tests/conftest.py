from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from wanderlust.wikivoyage.client import WikivoyageClient

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def examplestad_wikitext() -> str:
    return (FIXTURES / "examplestad.wikitext").read_text(encoding="utf-8")


@pytest.fixture
def fake_api(examplestad_wikitext) -> Callable[[httpx.Request], httpx.Response]:
    """A stand-in for the MediaWiki API that knows a single made-up article."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("action") == "parse":
            if params.get("page", "").casefold() in {"examplestad", "example town"}:
                return httpx.Response(
                    200,
                    json={
                        "parse": {
                            "title": "Examplestad",
                            "pageid": 42,
                            "revid": 7,
                            "wikitext": examplestad_wikitext,
                        }
                    },
                )
            return httpx.Response(
                200, json={"error": {"code": "missingtitle", "info": "doesn't exist"}}
            )
        if params.get("action") == "query":
            hits = [{"title": "Examplestad"}] if "example" in params["srsearch"].lower() else []
            return httpx.Response(200, json={"query": {"searchinfo": {}, "search": hits}})
        return httpx.Response(400)

    return handler


@pytest.fixture
def make_client(fake_api):
    """Build a WikivoyageClient wired to a mock transport and record its requests."""

    def factory(handler=None) -> tuple[WikivoyageClient, list[httpx.Request]]:
        requests: list[httpx.Request] = []

        def recording(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return (handler or fake_api)(request)

        http = httpx.Client(transport=httpx.MockTransport(recording))
        return WikivoyageClient("en", http_client=http), requests

    return factory
