from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from wanderlust.wikivoyage.client import WikivoyageClient

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def examplestad_wikitext() -> str:
    return (FIXTURES / "examplestad.wikitext").read_text(encoding="utf-8")


@pytest.fixture
def fake_api(examplestad_wikitext) -> Callable[[httpx.Request], httpx.Response]:
    """A stand-in for the MediaWiki API that knows a single made-up article."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("action") == "parse":
            if params.get("page", "").casefold() in {"examplestad", "example town"}:
                return httpx.Response(
                    200,
                    json={
                        "parse": {
                            "title": "Examplestad",
                            "pageid": 42,
                            "revid": 7,
                            "wikitext": examplestad_wikitext,
                        }
                    },
                )
            return httpx.Response(
                200, json={"error": {"code": "missingtitle", "info": "doesn't exist"}}
            )
        if params.get("action") == "query":
            hits = [{"title": "Examplestad"}] if "example" in params["srsearch"].lower() else []
            return httpx.Response(200, json={"query": {"searchinfo": {}, "search": hits}})
        return httpx.Response(400)

    return handler


@pytest.fixture
def make_client(fake_api):
    """Build a WikivoyageClient wired to a mock transport and record its requests."""

    def factory(handler=None) -> tuple[WikivoyageClient, list[httpx.Request]]:
        requests: list[httpx.Request] = []

        def recording(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return (handler or fake_api)(request)

        http = httpx.Client(transport=httpx.MockTransport(recording))
        return WikivoyageClient("en", http_client=http), requests

    return factory


class FakeEmbedder:
    """Deterministic offline embedder: hashed bag of words, so similar texts score higher."""

    name = "fake:bag-of-words"

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims
        self.calls: list[list[str]] = []

    def embed(self, texts):
        import hashlib

        import numpy as np

        from wanderlust.retrieval.embeddings import normalize
        from wanderlust.retrieval.keyword import tokenize

        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.md5(token.encode()).digest()
                vectors[row, int.from_bytes(digest[:4], "little") % self.dims] += 1.0
        return normalize(vectors)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
