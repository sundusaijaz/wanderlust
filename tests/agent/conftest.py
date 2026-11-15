from datetime import timedelta

import pytest

from wanderlust.agent.guides import GuideLibrary, SourceBook, index_opener
from wanderlust.retrieval.index import IndexStore
from wanderlust.wikivoyage.cache import ArticleCache
from wanderlust.wikivoyage.service import GuideLoader


@pytest.fixture
def guide_loader(make_client, tmp_path):
    client, requests = make_client()
    loader = GuideLoader(client, ArticleCache(tmp_path / "cache", ttl=timedelta(hours=1)))
    loader.requests = requests
    return loader


@pytest.fixture
def library(guide_loader, fake_embedder, tmp_path):
    store = IndexStore(tmp_path / "data")
    return GuideLibrary(index_opener(store, guide_loader, fake_embedder))


@pytest.fixture
def book():
    return SourceBook()
