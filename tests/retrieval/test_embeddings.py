import sys
import types

import httpx
import numpy as np
import pytest

from wanderlust.config import EmbeddingProvider, MissingAPIKeyError, Settings
from wanderlust.retrieval.embeddings import (
    Embedder,
    EmbeddingError,
    LocalEmbedder,
    OpenAIEmbedder,
    build_embedder,
    normalize,
)


def test_normalize_makes_unit_rows_and_keeps_zero_rows():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [0.6, 0.8])
    np.testing.assert_allclose(out[1], [0.0, 0.0])


def test_normalize_rejects_1d_input():
    with pytest.raises(ValueError):
        normalize(np.array([1.0, 2.0]))


def test_fake_embedder_satisfies_protocol(fake_embedder):
    assert isinstance(fake_embedder, Embedder)
    vectors = fake_embedder.embed(["fish soup", "fish soup restaurant", "tram line"])
    assert vectors.shape == (3, 64)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


def openai_client(handler):
    requests = []

    def recording(request):
        requests.append(request)
        return handler(request)

    http = httpx.Client(transport=httpx.MockTransport(recording))
    return OpenAIEmbedder("test-key", batch_size=2, http_client=http), requests


def echo_embeddings(request):
    """Return embeddings in reverse order to check that results are re-sorted by index."""
    import json

    inputs = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [float(len(text)), 1.0]} for i, text in enumerate(inputs)]
    return httpx.Response(200, json={"data": list(reversed(data))})


def test_openai_embedder_batches_and_orders_results():
    embedder, requests = openai_client(echo_embeddings)
    vectors = embedder.embed(["a", "bbb", "cc"])

    assert len(requests) == 2  # batch size 2
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert requests[0].url == "https://api.openai.com/v1/embeddings"
    expected = normalize(np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 1.0]]))
    np.testing.assert_allclose(vectors, expected, rtol=1e-6)
    assert embedder.name == "openai:text-embedding-3-small"


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "rejected the API key"), (429, "rate limit"), (500, "HTTP 500")],
)
def test_openai_embedder_http_errors(status, message):
    embedder, _ = openai_client(lambda r: httpx.Response(status))
    with pytest.raises(EmbeddingError, match=message):
        embedder.embed(["x"])


def test_openai_embedder_malformed_response():
    embedder, _ = openai_client(lambda r: httpx.Response(200, json={"nope": []}))
    with pytest.raises(EmbeddingError, match="Unexpected response"):
        embedder.embed(["x"])


def test_openai_embedder_count_mismatch():
    embedder, _ = openai_client(
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    )
    with pytest.raises(EmbeddingError, match="different number"):
        embedder.embed(["x", "y"])


def test_openai_embedder_network_error():
    def offline(request):
        raise httpx.ConnectError("offline", request=request)

    embedder, _ = openai_client(offline)
    with pytest.raises(EmbeddingError, match="Could not reach"):
        embedder.embed(["x"])


def test_embedders_reject_empty_input():
    embedder, _ = openai_client(echo_embeddings)
    with pytest.raises(ValueError):
        embedder.embed([])
    with pytest.raises(ValueError):
        LocalEmbedder().embed([])


def test_local_embedder_explains_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(EmbeddingError, match=r"wanderlust\[local\]"):
        LocalEmbedder().embed(["x"])


def test_local_embedder_uses_fastembed(monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            return [np.array([1.0, 1.0]) for _ in texts]

    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    )
    embedder = LocalEmbedder("tiny-model")
    vectors = embedder.embed(["a", "b"])
    assert embedder.name == "local:tiny-model"
    np.testing.assert_allclose(vectors, np.full((2, 2), 2**-0.5), rtol=1e-6)


@pytest.fixture
def settings_factory(monkeypatch):
    for name in ("OPENAI_API_KEY", "WANDERLUST_EMBEDDING_PROVIDER", "WANDERLUST_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def make(**kwargs):
        return Settings(_env_file=None, **kwargs)

    return make


def test_build_embedder_none(settings_factory):
    assert build_embedder(settings_factory(embedding_provider=EmbeddingProvider.NONE)) is None


def test_build_embedder_auto_prefers_openai_with_key(settings_factory, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = build_embedder(settings_factory(embedding_model="custom-model"))
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.name == "openai:custom-model"


def test_build_embedder_auto_falls_back_to_local(settings_factory, monkeypatch):
    monkeypatch.setattr(LocalEmbedder, "available", staticmethod(lambda: True))
    assert isinstance(build_embedder(settings_factory()), LocalEmbedder)


def test_build_embedder_auto_without_options_is_keyword_only(settings_factory, monkeypatch):
    monkeypatch.setattr(LocalEmbedder, "available", staticmethod(lambda: False))
    assert build_embedder(settings_factory()) is None


def test_build_embedder_openai_requires_key(settings_factory):
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        build_embedder(settings_factory(embedding_provider=EmbeddingProvider.OPENAI))


def test_build_embedder_local_explicit(settings_factory, monkeypatch):
    monkeypatch.setattr(LocalEmbedder, "available", staticmethod(lambda: False))
    embedder = build_embedder(settings_factory(embedding_provider=EmbeddingProvider.LOCAL))
    assert isinstance(embedder, LocalEmbedder)
