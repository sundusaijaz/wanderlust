from datetime import timedelta

import pytest

from wanderlust.retrieval.builder import build_index
from wanderlust.retrieval.search import Searcher
from wanderlust.wikivoyage.cache import ArticleCache
from wanderlust.wikivoyage.service import GuideLoader


@pytest.fixture
def examplestad_index(make_client, tmp_path, fake_embedder):
    client, _ = make_client()
    loader = GuideLoader(client, ArticleCache(tmp_path, ttl=timedelta(hours=1)))
    return build_index("Examplestad", loader, fake_embedder).index


@pytest.fixture
def searcher(examplestad_index, fake_embedder):
    return Searcher(examplestad_index, fake_embedder)
