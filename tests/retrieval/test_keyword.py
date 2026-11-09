import pytest

from wanderlust.retrieval.keyword import BM25Index, tokenize

DOCS = [
    "The old lighthouse has a viewpoint at the top. Climb the steps at sunset.",
    "Harbour Kitchen serves fish soup. Family-run restaurant near the harbour.",
    "Trams run from the station to the lighthouse every 15 minutes.",
    "The Salty Mug is a cosy pub by the pier.",
    "Three palaces and two museums sit on the hill above Belém.",
]


@pytest.fixture(scope="module")
def index():
    return BM25Index(DOCS)


def test_tokenize_lowercases_folds_accents_and_drops_stopwords():
    assert tokenize("The Museums of Belém, in 2024!") == ["museum", "belem", "2024"]


def test_tokenize_light_stemming():
    assert tokenize("palaces cities bus glass") == ["palace", "city", "bus", "glass"]


def test_best_match_ranks_first(index):
    results = index.search("fish soup restaurant")
    assert results[0][0] == 1
    assert all(score > 0 for _, score in results)


def test_rare_terms_outweigh_common_ones(index):
    # "lighthouse" appears twice in the corpus, "sunset" only once
    (top, _), *_ = index.search("lighthouse sunset")
    assert top == 0


def test_plural_and_accent_insensitive(index):
    assert index.search("palace belem")[0][0] == 4


def test_no_match_returns_empty(index):
    assert index.search("snorkelling") == []
    assert index.search("the and of") == []


def test_k_limits_results(index):
    assert len(index.search("the lighthouse harbour pub", k=2)) == 2


def test_repeated_query_terms_do_not_inflate_scores(index):
    assert index.scores("pub") == index.scores("pub pub pub")


def test_empty_index():
    empty = BM25Index([])
    assert len(empty) == 0
    assert empty.search("anything") == []
