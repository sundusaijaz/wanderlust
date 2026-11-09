import json

import numpy as np
import pytest

from wanderlust.retrieval.chunking import chunk_article
from wanderlust.retrieval.index import (
    GuideIndex,
    IndexLoadError,
    IndexMeta,
    IndexStore,
    VectorIndex,
)
from wanderlust.wikivoyage.parser import parse_article


@pytest.fixture
def chunks(examplestad_wikitext):
    article = parse_article(
        examplestad_wikitext, title="Examplestad", language="en", page_id=42, revision_id=7
    )
    return chunk_article(article)


def make_index(chunks, embedder=None) -> GuideIndex:
    vectors = embedder.embed([c.search_text for c in chunks]) if embedder else None
    meta = IndexMeta(
        destination="Examplestad",
        language="en",
        articles={"Examplestad": 7},
        embedder=embedder.name if embedder else None,
        dims=vectors.shape[1] if vectors is not None else None,
        chunk_count=len(chunks),
    )
    return GuideIndex(meta=meta, chunks=chunks, vectors=vectors)


def test_vector_index_returns_best_matches_first():
    index = VectorIndex(np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]]))
    results = index.search(np.array([1.0, 0.1]), k=2)
    assert [i for i, _ in results] == [0, 2]
    assert results[0][1] > results[1][1]


def test_vector_index_edge_cases():
    index = VectorIndex(np.eye(3))
    assert len(index.search(np.array([1.0, 1.0, 1.0]), k=10)) == 3
    assert index.search(np.zeros(3)) == []
    assert index.search(np.ones(3), k=0) == []
    with pytest.raises(ValueError, match="dimensions"):
        index.search(np.ones(2))
    with pytest.raises(ValueError):
        VectorIndex(np.ones(3))


def test_guide_index_validates_shapes(chunks, fake_embedder):
    index = make_index(chunks, fake_embedder)
    with pytest.raises(ValueError, match="chunk_count"):
        GuideIndex(meta=index.meta, chunks=chunks[:-1], vectors=index.vectors)
    with pytest.raises(ValueError, match="vectors"):
        GuideIndex(meta=index.meta, chunks=chunks, vectors=index.vectors[:, :10])


def test_bm25_is_built_from_search_text(chunks):
    index = make_index(chunks)
    top, _ = index.bm25.search("fish soup")[0]
    assert "fish soup" in index.chunks[top].text.lower()
    assert index.vector_index is None


def test_save_and_load_round_trip(tmp_path, chunks, fake_embedder):
    index = make_index(chunks, fake_embedder)
    index.save(tmp_path / "idx")
    loaded = GuideIndex.load(tmp_path / "idx")
    assert loaded.meta == index.meta
    assert loaded.chunks == index.chunks
    np.testing.assert_array_equal(loaded.vectors, index.vectors)


def test_keyword_only_index_round_trip(tmp_path, chunks, fake_embedder):
    path = tmp_path / "idx"
    make_index(chunks, fake_embedder).save(path)
    make_index(chunks).save(path)  # overwrite without vectors
    assert not (path / "vectors.npy").exists()
    assert GuideIndex.load(path).vectors is None


def test_load_rejects_other_format_versions(tmp_path, chunks):
    path = tmp_path / "idx"
    make_index(chunks).save(path)
    meta = json.loads((path / "meta.json").read_text())
    meta["format_version"] = 99
    (path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(IndexLoadError, match="not supported"):
        GuideIndex.load(path)


def test_load_reports_missing_and_corrupt_files(tmp_path, chunks):
    with pytest.raises(IndexLoadError, match="missing"):
        GuideIndex.load(tmp_path / "nothing")
    path = tmp_path / "idx"
    make_index(chunks).save(path)
    (path / "chunks.json").write_text("[{]")
    with pytest.raises(IndexLoadError, match="unreadable"):
        GuideIndex.load(path)


def test_store_round_trip_and_bad_entries(tmp_path, chunks, fake_embedder):
    store = IndexStore(tmp_path)
    assert store.load("en", "Examplestad") is None

    path = store.save(make_index(chunks, fake_embedder))
    assert path.parent == tmp_path / "index" / "en"
    assert store.load("en", "examplestad").meta.destination == "Examplestad"

    (path / "vectors.npy").write_bytes(b"garbage")
    assert store.load("en", "Examplestad") is None
