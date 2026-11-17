import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from wanderlust.config import Settings
from wanderlust.llm.fake import ScriptedProvider
from wanderlust.tools.http import OpenMeteoClient
from wanderlust.web.app import Services, create_app
from wanderlust.wikivoyage.cache import ArticleCache
from wanderlust.wikivoyage.service import GuideLoader


def meteo_api(request):
    if request.url.host.startswith("geocoding"):
        name = request.url.params["name"]
        if name.casefold() != "examplestad":
            return httpx.Response(200, json={})
        place = {
            "name": "Examplestad",
            "latitude": 12.34,
            "longitude": -7.65,
            "country": "Imaginaria",
        }
        return httpx.Response(200, json={"results": [place]})
    return httpx.Response(500)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", openai_api_key="test-key")


@pytest.fixture
def script():
    """The model's replies for the next plan; tests fill it in."""
    return []


@pytest.fixture
def services(make_client, fake_embedder, tmp_path, script):
    client, _ = make_client()
    cache = ArticleCache(tmp_path / "cache", ttl=timedelta(hours=1))

    def tools_client(data_dir):
        return OpenMeteoClient(
            http_client=httpx.Client(transport=httpx.MockTransport(meteo_api)),
            clock=lambda: datetime(2026, 11, 17, 9, tzinfo=UTC),
        )

    return Services(
        provider=lambda settings: ScriptedProvider(script),
        embedder=lambda settings: fake_embedder,
        loader=lambda settings: GuideLoader(client, cache),
        tools_client=tools_client,
    )


@pytest.fixture
def make_web(settings, services):
    def factory(**options):
        return TestClient(create_app(settings, services, **options))

    return factory


@pytest.fixture
def web(make_web):
    return make_web()


def parse_events(body: str) -> list[tuple[str, dict]]:
    """Split a server-sent event stream into (event, data) pairs, skipping comments."""
    events = []
    for block in body.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if not line.startswith(":")]
        if not lines:
            continue
        fields = dict(line.split(": ", 1) for line in lines)
        events.append((fields["event"], json.loads(fields["data"])))
    return events
