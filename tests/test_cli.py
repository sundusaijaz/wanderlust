import pytest
from typer.testing import CliRunner

from wanderlust import __version__
from wanderlust.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Run each test in an empty folder so a developer's .env is not picked up."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")  # wide enough that tables don't wrap mid-word
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "WANDERLUST_LLM_PROVIDER",
        "WANDERLUST_EMBEDDING_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_config_reports_missing_key():
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "missing" in result.output


def test_config_never_prints_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "configured" in result.output
    assert "test-secret-value" not in result.output


def test_config_reports_invalid_settings(monkeypatch):
    monkeypatch.setenv("WANDERLUST_LLM_PROVIDER", "not-a-provider")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.output


@pytest.fixture
def fake_guides(monkeypatch, make_client, tmp_path):
    """Point the guide command at the made-up Examplestad API and a temp cache."""
    from datetime import timedelta

    from wanderlust.wikivoyage.cache import ArticleCache
    from wanderlust.wikivoyage.service import GuideLoader

    client, requests = make_client()
    cache = ArticleCache(tmp_path / "cache", ttl=timedelta(hours=1))
    monkeypatch.setattr("wanderlust.cli.build_loader", lambda settings: GuideLoader(client, cache))
    return requests


def test_guide_overview(fake_guides):
    result = runner.invoke(app, ["guide", "Examplestad"])
    assert result.exit_code == 0, result.output
    assert "fictional harbour town" in result.output
    for section in ("Understand", "By tram", "See", "Eat", "Go next"):
        assert section in result.output
    assert "https://en.wikivoyage.org/wiki/Examplestad" in result.output
    assert "CC BY-SA 4.0" in result.output
    assert "downloaded just now" in result.output


def test_guide_second_call_uses_cache(fake_guides):
    runner.invoke(app, ["guide", "Examplestad"])
    result = runner.invoke(app, ["guide", "examplestad"])
    assert "from cache" in result.output
    assert len(fake_guides) == 1


def test_guide_refresh_downloads_again(fake_guides):
    runner.invoke(app, ["guide", "Examplestad"])
    result = runner.invoke(app, ["guide", "Examplestad", "--refresh"])
    assert "downloaded just now" in result.output
    assert len(fake_guides) == 2


def test_guide_section_shows_listings(fake_guides):
    result = runner.invoke(app, ["guide", "Examplestad", "--section", "see"])
    assert result.exit_code == 0, result.output
    assert "Old Lighthouse" in result.output
    assert "Vieux Phare" in result.output
    assert "Adult 5 coins" in result.output
    assert "Examplestad#See" in result.output


def test_guide_section_includes_subsections(fake_guides):
    result = runner.invoke(app, ["guide", "Examplestad", "-s", "Get around"])
    assert "By tram" in result.output
    assert "heritage tram line" in result.output
    assert "By bicycle" in result.output


def test_guide_unknown_section_lists_alternatives(fake_guides):
    result = runner.invoke(app, ["guide", "Examplestad", "-s", "Nightlife"])
    assert result.exit_code == 1
    assert "No section called 'Nightlife'" in result.output
    assert "Understand, Get around, See, Eat, Go next" in result.output


def test_guide_unknown_place_suggests_alternative(fake_guides):
    result = runner.invoke(app, ["guide", "Example harbour"])
    assert result.exit_code == 1
    assert "Did you mean 'Examplestad'" in result.output
    assert 'wanderlust guide "Examplestad"' in result.output


def test_guide_network_error(monkeypatch, make_client, tmp_path):
    import httpx

    from wanderlust.wikivoyage.cache import ArticleCache
    from wanderlust.wikivoyage.service import GuideLoader

    def offline(request):
        raise httpx.ConnectError("offline", request=request)

    client, _ = make_client(offline)
    loader = GuideLoader(client, ArticleCache(tmp_path))
    monkeypatch.setattr("wanderlust.cli.build_loader", lambda settings: loader)
    result = runner.invoke(app, ["guide", "Examplestad"])
    assert result.exit_code == 1
    assert "Could not reach Wikivoyage" in result.output


@pytest.fixture
def semantic(monkeypatch, fake_embedder):
    """Use the deterministic fake embedder instead of OpenAI or a local model."""
    monkeypatch.setattr("wanderlust.cli.build_embedder", lambda settings: fake_embedder)
    return fake_embedder


@pytest.fixture
def keyword_only(monkeypatch):
    monkeypatch.setattr("wanderlust.cli.build_embedder", lambda settings: None)


def test_index_reports_passages_and_districts(fake_guides, semantic):
    result = runner.invoke(app, ["index", "Bigville"])
    assert result.exit_code == 0, result.output
    assert "Indexed Bigville" in result.output
    assert "from 2 articles" in result.output
    assert "Districts: Old Town" in result.output
    assert "Skipped: Bigville/Docks" in result.output
    assert "fake:bag-of-words (64 dimensions)" in result.output


def test_index_without_districts(fake_guides, keyword_only):
    result = runner.invoke(app, ["index", "Bigville", "--no-districts"])
    assert result.exit_code == 0, result.output
    assert "from 1 article." in result.output
    assert "Keyword search only" in result.output
    assert "wanderlust[local]" in result.output


def test_index_unknown_place(fake_guides, keyword_only):
    result = runner.invoke(app, ["index", "Example harbour"])
    assert result.exit_code == 1
    assert 'wanderlust index "Examplestad"' in result.output


def test_search_builds_index_on_first_use_then_reuses_it(fake_guides, semantic):
    first = runner.invoke(app, ["search", "Examplestad", "fish soup"])
    assert first.exit_code == 0, first.output
    assert "building it now" in first.output
    assert "Harbour Kitchen" in first.output or "Fish soup" in first.output
    assert "https://en.wikivoyage.org/wiki/Examplestad#Eat" in first.output
    assert "hybrid search over" in first.output

    second = runner.invoke(app, ["search", "examplestad", "lighthouse"])
    assert "building it now" not in second.output
    assert "Old Lighthouse" in second.output


def test_search_shows_ranks_and_respects_top(fake_guides, semantic):
    result = runner.invoke(app, ["search", "Examplestad", "lighthouse sunset", "-k", "1"])
    assert result.exit_code == 0, result.output
    assert "1. Examplestad > See · Old Lighthouse" in result.output
    assert "keyword #1" in result.output and "vector #" in result.output
    assert "2. " not in result.output


def test_search_keyword_only_explains_fallback(fake_guides, keyword_only):
    result = runner.invoke(app, ["search", "Examplestad", "tram"])
    assert result.exit_code == 0, result.output
    assert "Using keyword search only" in result.output
    assert "keyword search over" in result.output


def test_search_vector_mode_unavailable(fake_guides, keyword_only):
    result = runner.invoke(app, ["search", "Examplestad", "tram", "--mode", "vector"])
    assert result.exit_code == 1
    assert "Vector search is unavailable" in result.output


def test_search_without_matches(fake_guides, keyword_only):
    result = runner.invoke(app, ["search", "Examplestad", "snorkelling"])
    assert result.exit_code == 1
    assert "No passages about 'snorkelling'" in result.output


def test_search_missing_openai_key(fake_guides, monkeypatch):
    monkeypatch.setenv("WANDERLUST_EMBEDDING_PROVIDER", "openai")
    result = runner.invoke(app, ["search", "Examplestad", "tram"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_config_shows_semantic_search_choice(monkeypatch):
    from wanderlust.retrieval.embeddings import LocalEmbedder

    monkeypatch.setattr(LocalEmbedder, "available", staticmethod(lambda: False))
    result = runner.invoke(app, ["config"])
    assert "Semantic search" in result.output
    assert "keyword search only" in result.output

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")
    result = runner.invoke(app, ["config"])
    assert "openai:text-embedding-3-small" in result.output


@pytest.fixture
def scripted_llm(monkeypatch):
    from wanderlust.llm.fake import ScriptedProvider

    provider = ScriptedProvider(["Yes, Lisbon is very hilly.\nWear comfy [shoes]."])
    monkeypatch.setattr("wanderlust.cli.build_provider", lambda settings: provider)
    return provider


def test_chat_streams_the_answer(scripted_llm):
    result = runner.invoke(app, ["chat", "Is Lisbon hilly?"])
    assert result.exit_code == 0, result.output
    assert "fake · scripted" in result.output
    assert "Yes, Lisbon is very hilly.\nWear comfy [shoes]." in result.output

    conversation, tools = scripted_llm.requests[0]
    assert "Wanderlust" in conversation.system
    assert conversation.turns[0].text == "Is Lisbon hilly?"
    assert tools == []


def test_chat_without_api_key():
    result = runner.invoke(app, ["chat", "hello"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_chat_reports_provider_errors(monkeypatch):
    from wanderlust.llm.fake import ScriptedProvider

    monkeypatch.setattr("wanderlust.cli.build_provider", lambda settings: ScriptedProvider([]))
    result = runner.invoke(app, ["chat", "hello"])
    assert result.exit_code == 1
    assert "no responses left" in result.output


def test_config_shows_default_model_and_effort(monkeypatch):
    monkeypatch.setenv("WANDERLUST_LLM_PROVIDER", "anthropic")
    result = runner.invoke(app, ["config"])
    assert "claude-opus-5-5 (default)" in result.output
    assert "LLM effort" in result.output


def use_llm(monkeypatch, *script):
    from wanderlust.llm.fake import ScriptedProvider

    provider = ScriptedProvider(script)
    monkeypatch.setattr("wanderlust.cli.build_provider", lambda settings: provider)
    return provider


def test_ask_streams_answer_and_lists_cited_sources(fake_guides, semantic, monkeypatch):
    llm = use_llm(monkeypatch, "Climb the Old Lighthouse at sunset [1]. Entry is 5 coins [1].")
    result = runner.invoke(app, ["ask", "Examplestad", "lighthouse at sunset"])
    assert result.exit_code == 0, result.output
    assert "fake · scripted" in result.output
    assert "Climb the Old Lighthouse at sunset [1]." in result.output
    assert "Sources" in result.output
    assert "[1] Examplestad > See · Old Lighthouse" in result.output
    assert "https://en.wikivoyage.org/wiki/Examplestad#See" in result.output
    assert "CC BY-SA 4.0" in result.output

    conversation, _ = llm.requests[0]
    assert "Question: lighthouse at sunset" in conversation.turns[0].text


def test_ask_warns_about_invented_citations(fake_guides, semantic, monkeypatch):
    use_llm(monkeypatch, "Fish soup at Harbour Kitchen [1], and a beach bar [12].")
    result = runner.invoke(app, ["ask", "Examplestad", "fish soup"])
    assert result.exit_code == 0, result.output
    assert "cites [12], which is not one of the passages" in result.output


def test_ask_answer_without_citations_lists_given_passages(fake_guides, semantic, monkeypatch):
    use_llm(monkeypatch, "Fish soup is the local speciality.")
    result = runner.invoke(app, ["ask", "Examplestad", "fish soup", "-k", "2"])
    assert "The answer cites no passages" in result.output
    assert "[2] " in result.output


def test_ask_show_passages(fake_guides, semantic, monkeypatch):
    use_llm(monkeypatch, "Trams run every 15 minutes [1].")
    result = runner.invoke(app, ["ask", "Examplestad", "tram", "--show-passages", "-k", "2"])
    assert result.exit_code == 0, result.output
    assert "[1] Examplestad" in result.output
    assert "heritage tram line" in result.output


def test_ask_nothing_retrieved_does_not_call_the_model(fake_guides, keyword_only, monkeypatch):
    llm = use_llm(monkeypatch)
    result = runner.invoke(app, ["ask", "Examplestad", "snorkelling"])
    assert result.exit_code == 1
    assert "nothing to answer from" in result.output
    assert llm.requests == []


def test_ask_requires_an_llm_key(fake_guides, keyword_only):
    result = runner.invoke(app, ["ask", "Examplestad", "tram"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_ask_reports_llm_errors(fake_guides, semantic, monkeypatch):
    use_llm(monkeypatch)  # empty script -> error on first call
    result = runner.invoke(app, ["ask", "Examplestad", "tram"])
    assert result.exit_code == 1
    assert "no responses left" in result.output


@pytest.fixture
def fake_open_meteo(monkeypatch):
    """Serve made-up places and weather, with 'today' fixed at 2026-11-12."""
    from datetime import UTC, date, datetime

    import httpx

    from wanderlust.tools.http import OpenMeteoClient

    places = {
        "examplestad": {
            "name": "Examplestad",
            "latitude": 12.34,
            "longitude": -7.65,
            "country": "Imaginaria",
        },
        "otherville": {
            "name": "Otherville",
            "latitude": 12.5,
            "longitude": -7.5,
            "country": "Imaginaria",
        },
    }

    def api(request):
        params = request.url.params
        if request.url.host.startswith("geocoding"):
            match = places.get(params["name"].casefold())
            return httpx.Response(200, json={"results": [match]} if match else {})
        start, end = (
            date.fromisoformat(params["start_date"]),
            date.fromisoformat(params["end_date"]),
        )
        days = [
            date.fromordinal(d).isoformat() for d in range(start.toordinal(), end.toordinal() + 1)
        ]
        n = len(days)
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": days,
                    "weather_code": ([0, 63] * n)[:n],
                    "temperature_2m_max": [20.4] * n,
                    "temperature_2m_min": [12.6] * n,
                    "precipitation_sum": ([0.0, 9.0] * n)[:n],
                    "precipitation_probability_max": ([0, 85] * n)[:n],
                }
            },
        )

    def build(data_dir):
        return OpenMeteoClient(
            http_client=httpx.Client(transport=httpx.MockTransport(api)),
            clock=lambda: datetime(2026, 11, 12, 9, tzinfo=UTC),
        )

    monkeypatch.setattr("wanderlust.cli.build_tools_client", build)


def test_weather_forecast_table(fake_open_meteo):
    result = runner.invoke(app, ["weather", "Examplestad", "--days", "2"])
    assert result.exit_code == 0, result.output
    assert "Forecast for Examplestad, Imaginaria" in result.output
    assert "Thu 12 Nov" in result.output and "Fri 13 Nov" in result.output
    assert "Clear sky" in result.output and "Rain" in result.output
    assert "13 to 20" in result.output
    assert "indoor day" in result.output
    assert "Open-Meteo" in result.output


def test_weather_typical_for_later_dates(fake_open_meteo):
    result = runner.invoke(app, ["weather", "Examplestad", "--from", "2027-03-01", "-d", "3"])
    assert result.exit_code == 0, result.output
    assert "Typical weather for Examplestad" in result.output
    assert "not a forecast" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["weather", "Nowhereville"], "No place called 'Nowhereville'"),
        (["weather", "Examplestad", "--from", "March"], "Invalid date"),
        (["weather", "Examplestad", "--from", "2026-01-01"], "past"),
    ],
)
def test_weather_errors(fake_open_meteo, args, message):
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert message in result.output


def test_distance_between_places(fake_open_meteo):
    result = runner.invoke(app, ["distance", "Examplestad", "Otherville"])
    assert result.exit_code == 0, result.output
    assert "Examplestad, Imaginaria → Otherville, Imaginaria" in result.output
    assert "km in a straight line" in result.output
    assert "Public transport: about" in result.output
    assert "Rough estimates" in result.output


def test_distance_unknown_place(fake_open_meteo):
    result = runner.invoke(app, ["distance", "Examplestad", "Atlantis"])
    assert result.exit_code == 1
    assert "No place called 'Atlantis'" in result.output


# -- plan -----------------------------------------------------------------------


def _call(name, n=1, **arguments):
    from wanderlust.llm.types import ToolCall

    return ToolCall(id=f"{name}-{n}", name=name, arguments=arguments)


def _plan_script(**overrides):
    from wanderlust.llm.fake import ScriptedProvider

    itinerary = {
        "title": "Two days in Examplestad",
        "summary": "Lighthouse walks and fish soup.",
        "days": [
            {
                "day": 1,
                "date": "2026-11-13",
                "base": "Examplestad",
                "title": "The lighthouse",
                "weather": "Clear sky, up to 20 C",
                "stops": [
                    {
                        "time_of_day": "evening",
                        "name": "Old Lighthouse",
                        "description": "Climb the 120 steps at sunset.",
                        "sources": [1],
                    }
                ],
                "getting_around": "Heritage tram to the end of the line.",
            },
            {
                "day": 2,
                "date": "2026-11-14",
                "base": "Examplestad",
                "title": "A rainy harbour day",
                "weather": "Rain",
                "stops": [
                    {
                        "time_of_day": "afternoon",
                        "name": "Harbour Kitchen",
                        "description": "Family-run [cheap] fish soup.",
                        "sources": [2, 1],
                        "indoor": True,
                    }
                ],
            },
        ],
        "tips": ["Trams run every 15 minutes [2]."],
        "assumptions": ["Trip taken as 13-14 November 2026."],
        **overrides,
    }
    return [
        ScriptedProvider.tool_request(
            _call("search_guide", destination="Examplestad", query="lighthouse fish soup tram"),
            text="Let me research the guide first.",
        ),
        ScriptedProvider.tool_request(
            _call("geocode", place="Examplestad"),
            _call(
                "get_weather",
                latitude=12.34,
                longitude=-7.65,
                start_date="2026-11-13",
                end_date="2026-11-14",
            ),
        ),
        ScriptedProvider.tool_request(_call("submit_itinerary", **itinerary)),
    ]


def test_plan_prints_a_cited_itinerary(fake_guides, semantic, fake_open_meteo, monkeypatch):
    llm = use_llm(monkeypatch, *_plan_script())
    result = runner.invoke(app, ["plan", "Two days in Examplestad"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "fake · scripted · planning: Two days in Examplestad" in out
    assert "Two days in Examplestad" in out
    assert "Day 1 · Fri 13 Nov 2026 · Examplestad" in out
    assert "Weather: Clear sky, up to 20 C" in out
    assert "Evening" in out and "Old Lighthouse [1]" in out
    assert "Harbour Kitchen [2][1]" in out and "indoor" in out
    assert "Family-run [cheap] fish soup." in out  # model text is not read as markup
    assert "Trams run every 15 minutes [2]." in out
    assert "Trip taken as 13-14 November 2026." in out
    assert "Sources" in out and "[1] Examplestad" in out
    assert "https://en.wikivoyage.org/wiki/Examplestad" in out
    assert "3 model calls, 4 tool calls" in out
    assert "CC BY-SA 4.0" in out and "Open-Meteo" in out
    assert "research the guide first" not in out  # only shown with --verbose
    assert llm.remaining == 0


def test_plan_verbose_shows_tool_calls(fake_guides, semantic, fake_open_meteo, monkeypatch):
    use_llm(monkeypatch, *_plan_script())
    result = runner.invoke(app, ["plan", "Two days in Examplestad", "--verbose"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Let me research the guide first." in out
    assert "Searching the Examplestad guide for 'lighthouse fish soup tram'" in out
    assert "→ 5 passages" in out
    assert "Looking up Examplestad → Examplestad, Imaginaria" in out
    assert "forecast, 2 days, 1 wet" in out
    assert "Checking the itinerary → accepted" in out


def test_plan_json(fake_guides, semantic, fake_open_meteo, monkeypatch):
    import json

    use_llm(monkeypatch, *_plan_script())
    result = runner.invoke(app, ["plan", "Two days in Examplestad", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["request"] == "Two days in Examplestad"
    assert payload["itinerary"]["days"][1]["stops"][0]["indoor"] is True
    assert [s["number"] for s in payload["sources"]] == [1, 2]
    assert payload["sources"][0]["url"].startswith("https://en.wikivoyage.org/wiki/Examplestad")
    assert payload["model"] == {"provider": "fake", "name": "scripted"}


def test_plan_failure_is_reported(fake_guides, semantic, fake_open_meteo, monkeypatch):
    use_llm(monkeypatch, "I would need more details.", "Sorry, I can't.")
    result = runner.invoke(app, ["plan", "Somewhere nice"])
    assert result.exit_code == 1
    assert "stopped without submitting an itinerary" in result.output
    assert "The model said: Sorry, I can't." in result.output
    assert "2 model calls, 0 tool calls" in result.output


def test_plan_step_limit_option(fake_guides, semantic, fake_open_meteo, monkeypatch):
    use_llm(monkeypatch, *_plan_script())
    result = runner.invoke(app, ["plan", "Two days in Examplestad", "--max-steps", "2"])
    assert result.exit_code == 1
    assert "No itinerary after 2 steps" in result.output


def test_plan_without_api_key(fake_guides, keyword_only):
    result = runner.invoke(app, ["plan", "Two days in Examplestad"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_plan_reports_provider_errors(fake_guides, semantic, fake_open_meteo, monkeypatch):
    use_llm(monkeypatch)  # no responses: the first model call fails
    result = runner.invoke(app, ["plan", "Two days in Examplestad"])
    assert result.exit_code == 1
    assert "no responses left" in result.output
