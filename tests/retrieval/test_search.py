from datetime import timedelta

import pytest

from wanderlust.retrieval.builder import build_index
from wanderlust.retrieval.search import (
    Searcher,
    SearchError,
    SearchMode,
    reciprocal_rank_fusion,
)
from wanderlust.wikivoyage.cache import ArticleCache
from wanderlust.wikivoyage.service import GuideLoader


@pytest.fixture
def loader(make_client, tmp_path):
    client, _ = make_client()
    return GuideLoader(client, ArticleCache(tmp_path, ttl=timedelta(hours=1)))


@pytest.fixture
def report(loader, fake_embedder):
    return build_index("Examplestad", loader, fake_embedder)


def test_rrf_rewards_items_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert [item for item, _ in fused] == [1, 3, 2, 4]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_ties_break_by_item_for_stable_output():
    assert [i for i, _ in reciprocal_rank_fusion([[5], [2]])] == [2, 5]


def test_build_index_records_what_was_used(report, fake_embedder):
    meta = report.index.meta
    assert meta.destination == "Examplestad"
    assert meta.articles == {"Examplestad": 7}
    assert meta.embedder == fake_embedder.name
    assert meta.dims == 64
    assert meta.chunk_count == len(report.index.chunks) > 5
    assert len(fake_embedder.calls) == 1  # all chunks embedded in one call


def test_build_index_without_embedder_is_keyword_only(loader):
    index = build_index("Examplestad", loader, None).index
    assert index.vectors is None and index.meta.embedder is None


def test_build_index_reports_progress(loader, fake_embedder):
    messages: list[str] = []
    build_index("Examplestad", loader, fake_embedder, on_progress=messages.append)
    assert messages[0] == "Loading the guide for Examplestad"
    assert messages[-1].startswith("Embedding")


@pytest.mark.parametrize("mode", list(SearchMode))
def test_every_mode_finds_the_fish_soup(report, fake_embedder, mode):
    response = Searcher(report.index, fake_embedder).search("fish soup", k=3, mode=mode)
    assert response.mode is mode
    assert response.note is None
    assert any("fish soup" in r.chunk.text.lower() for r in response.results)


def test_hybrid_results_carry_both_ranks(report, fake_embedder):
    response = Searcher(report.index, fake_embedder).search("lighthouse sunset steps")
    top = response.results[0]
    assert top.chunk.listing_name == "Old Lighthouse"
    assert top.keyword_rank is not None and top.vector_rank is not None


def test_k_limits_results(report, fake_embedder):
    assert len(Searcher(report.index, fake_embedder).search("the town", k=2).results) <= 2


def test_hybrid_falls_back_to_keyword_without_embedder(report):
    response = Searcher(report.index, None).search("tram")
    assert response.mode is SearchMode.KEYWORD
    assert "no embedding provider" in response.note
    assert response.results[0].chunk.section == "By tram"


def test_embedder_mismatch_is_explained(report):
    class OtherEmbedder:
        name = "other:model"

        def embed(self, texts):
            raise AssertionError("must not be called")

    searcher = Searcher(report.index, OtherEmbedder())
    response = searcher.search("tram")
    assert response.mode is SearchMode.KEYWORD
    assert "rebuild" in response.note
    with pytest.raises(SearchError, match="built with fake:bag-of-words"):
        searcher.search("tram", mode=SearchMode.VECTOR)


def test_vector_mode_on_keyword_only_index(loader):
    index = build_index("Examplestad", loader, None).index
    with pytest.raises(SearchError, match="without embeddings"):
        Searcher(index, None).search("tram", mode=SearchMode.VECTOR)


def test_invalid_queries(report):
    searcher = Searcher(report.index)
    with pytest.raises(ValueError):
        searcher.search("   ")
    with pytest.raises(ValueError):
        searcher.search("tram", k=0)


def test_no_keyword_match_returns_nothing(report):
    assert Searcher(report.index).search("snorkelling").results == []


def test_build_index_follows_districts(loader, fake_embedder):
    report = build_index("Bigville", loader, fake_embedder)
    # "Old Quarter" redirects to "Old Town", which must only be indexed once.
    assert [a.title for a in report.articles] == ["Bigville", "Bigville/Old Town"]
    assert report.skipped == ["Bigville/Docks"]
    ids = [chunk.id for chunk in report.index.chunks]
    assert len(ids) == len(set(ids))
    assert report.index.meta.articles == {"Bigville": 7, "Bigville/Old Town": 7}

    top = Searcher(report.index, fake_embedder).search("cathedral bell tower").results[0]
    assert top.chunk.listing_name == "Bigville Cathedral"
    assert top.chunk.breadcrumb == "Bigville/Old Town > See"


def test_build_index_can_skip_districts(loader):
    report = build_index("Bigville", loader, None, follow_districts=False)
    assert [a.title for a in report.articles] == ["Bigville"]
    assert report.skipped == []


def test_build_index_caps_district_count(loader):
    report = build_index("Bigville", loader, None, max_districts=1)
    assert [a.title for a in report.articles] == ["Bigville", "Bigville/Old Town"]
    assert report.skipped == ["Bigville/Docks", "Bigville/Old Quarter"]
