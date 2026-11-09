from datetime import timedelta

import pytest

from wanderlust.config import Settings
from wanderlust.wikivoyage.cache import ArticleCache, normalize_title
from wanderlust.wikivoyage.parser import parse_article
from wanderlust.wikivoyage.service import GuideLoader, build_loader


@pytest.fixture
def article(examplestad_wikitext):
    return parse_article(
        examplestad_wikitext, title="Examplestad", language="en", page_id=42, revision_id=7
    )


@pytest.fixture
def cache(tmp_path):
    return ArticleCache(tmp_path, ttl=timedelta(hours=1))


def test_normalize_title():
    assert normalize_title(" New_York  City ") == normalize_title("new york city")


def test_round_trip(cache, article):
    assert cache.get("en", "Examplestad") is None
    cache.put(article)
    assert cache.get("en", "examplestad") == article


def test_entries_are_per_language(cache, article):
    cache.put(article)
    assert cache.get("de", "Examplestad") is None


def test_requested_title_is_cached_too(cache, article):
    cache.put(article, requested_title="Example town")
    assert cache.get("en", "example town").title == "Examplestad"


def test_stale_entries_are_ignored(cache, article):
    cache.put(article)
    later = article.fetched_at + timedelta(hours=2)
    assert cache.get("en", "Examplestad", now=later) is None


def test_corrupt_entries_are_ignored(cache, article):
    cache.put(article)
    cache.path_for("en", "Examplestad").write_text("{not json", encoding="utf-8")
    assert cache.get("en", "Examplestad") is None


def test_paths_are_safe_and_distinct(cache):
    weird = cache.path_for("en", "../../etc/passwd")
    assert weird.parent == cache.root / "en"
    assert cache.path_for("en", "São Paulo") != cache.path_for("en", "So Paulo")


def test_clear(cache, article):
    cache.put(article, requested_title="Example town")
    assert cache.clear() == 2
    assert cache.get("en", "Examplestad") is None


def test_loader_uses_cache_after_first_fetch(make_client, cache):
    client, requests = make_client()
    loader = GuideLoader(client, cache)

    first = loader.load("Examplestad")
    second = loader.load("examplestad")

    assert not first.from_cache
    assert second.from_cache
    assert second.article == first.article
    assert len(requests) == 1


def test_loader_refresh_bypasses_cache(make_client, cache):
    client, requests = make_client()
    loader = GuideLoader(client, cache)
    loader.load("Examplestad")
    assert not loader.load("Examplestad", refresh=True).from_cache
    assert len(requests) == 2


def test_build_loader_follows_settings(tmp_path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        wikivoyage_language="fr",
        wikivoyage_cache_ttl_hours=3,
    )
    loader = build_loader(settings)
    try:
        assert loader.client.language == "fr"
        assert loader.cache.root == tmp_path / "wikivoyage"
        assert loader.cache.ttl == timedelta(hours=3)
    finally:
        loader.close()
