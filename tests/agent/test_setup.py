from datetime import UTC, date, datetime

import httpx

from wanderlust.agent.setup import build_agent
from wanderlust.config import Settings
from wanderlust.llm.fake import ScriptedProvider
from wanderlust.tools.http import OpenMeteoClient


def test_build_agent_wires_guides_tools_and_clock(guide_loader, fake_embedder, tmp_path):
    client = OpenMeteoClient(
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
        clock=lambda: datetime(2027, 3, 1, 8, tzinfo=UTC),
    )
    built = []
    agent = build_agent(
        Settings(data_dir=tmp_path / "data", _env_file=None),
        ScriptedProvider([]),
        loader=guide_loader,
        embedder=fake_embedder,
        tools_client=client,
        max_steps=5,
        on_build=built.append,
    )
    assert agent.today() == date(2027, 3, 1)
    assert agent.max_steps == 5
    assert agent.travel_tools.names == ["geocode", "get_weather", "get_distance"]

    agent.library.searcher("Examplestad")
    assert built == ["Examplestad"]
    assert (tmp_path / "data" / "index" / "en").is_dir()
