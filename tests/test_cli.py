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


def test_plan_is_a_placeholder_for_now():
    result = runner.invoke(app, ["plan", "4 days in Lisbon"])
    assert result.exit_code == 2
    assert "4 days in Lisbon" in result.output


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
