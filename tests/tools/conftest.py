from datetime import UTC, datetime, timedelta

import httpx
import pytest

from wanderlust.tools.http import JsonCache, OpenMeteoClient


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock():
    return FakeClock(datetime(2026, 11, 12, 9, 0, tzinfo=UTC))


@pytest.fixture
def make_meteo(tmp_path, clock):
    """An OpenMeteoClient wired to a handler function, recording every request."""

    def factory(handler, *, cache=True):
        requests: list[httpx.Request] = []

        def recording(request):
            requests.append(request)
            return handler(request)

        client = OpenMeteoClient(
            cache=JsonCache(tmp_path / "http", clock=clock) if cache else None,
            http_client=httpx.Client(transport=httpx.MockTransport(recording)),
            clock=clock,
        )
        return client, requests

    return factory
