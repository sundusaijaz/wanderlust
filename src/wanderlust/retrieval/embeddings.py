"""Turn text into vectors for semantic search."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np

from wanderlust.config import EmbeddingProvider, MissingAPIKeyError, Settings

OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
LOCAL_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingError(RuntimeError):
    """Embeddings could not be created."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps texts to L2-normalised vectors of a fixed size."""

    @property
    def name(self) -> str:
        """Identifies provider and model; stored with an index to detect mismatches."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dims)`` with unit-length rows."""
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("expected a 2-D array of vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class OpenAIEmbedder:
    """Embeddings from the OpenAI API, sent in batches."""

    endpoint = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = OPENAI_DEFAULT_MODEL,
        batch_size: int = 96,
        http_client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._http = http_client or httpx.Client(
            timeout=timeout, transport=httpx.HTTPTransport(retries=2)
        )
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("nothing to embed")
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            rows.extend(self._embed_batch(list(texts[start : start + self.batch_size])))
        return normalize(np.array(rows))

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._http.post(
                self.endpoint,
                headers=self._headers,
                json={"model": self.model, "input": batch},
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Could not reach the OpenAI API: {exc}") from exc

        if response.status_code == 401:
            raise EmbeddingError("OpenAI rejected the API key (HTTP 401).")
        if response.status_code == 429:
            raise EmbeddingError("OpenAI rate limit reached (HTTP 429). Try again shortly.")
        if response.is_error:
            raise EmbeddingError(f"OpenAI embeddings request failed (HTTP {response.status_code}).")

        try:
            data: list[dict[str, Any]] = response.json()["data"]
            ordered = sorted(data, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("Unexpected response from the OpenAI embeddings API") from exc
        if len(vectors) != len(batch):
            raise EmbeddingError("OpenAI returned a different number of embeddings than requested")
        return vectors


class LocalEmbedder:
    """Embeddings computed on this machine with fastembed (no API key needed).

    Install with ``pip install "wanderlust[local]"``. The model is downloaded
    once on first use.
    """

    def __init__(self, model: str = LOCAL_DEFAULT_MODEL) -> None:
        self.model = model
        self._engine: Any = None

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("fastembed") is not None

    @property
    def name(self) -> str:
        return f"local:{self.model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("nothing to embed")
        if self._engine is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise EmbeddingError(
                    'Local embeddings need fastembed: pip install "wanderlust[local]"'
                ) from exc
            self._engine = TextEmbedding(model_name=self.model)
        return normalize(np.array(list(self._engine.embed(list(texts)))))


def build_embedder(settings: Settings) -> Embedder | None:
    """Choose an embedder from settings, or ``None`` for keyword-only search.

    Raises:
        MissingAPIKeyError: if OpenAI embeddings are requested without a key.
    """
    provider = settings.embedding_provider
    openai_key = (
        settings.openai_api_key.get_secret_value().strip() if settings.openai_api_key else ""
    )

    if provider is EmbeddingProvider.NONE:
        return None
    if provider is EmbeddingProvider.OPENAI or (provider is EmbeddingProvider.AUTO and openai_key):
        if not openai_key:
            raise MissingAPIKeyError(
                "OpenAI embeddings need OPENAI_API_KEY. Set it, or use "
                "WANDERLUST_EMBEDDING_PROVIDER=local or none."
            )
        return OpenAIEmbedder(openai_key, model=settings.embedding_model or OPENAI_DEFAULT_MODEL)
    if provider is EmbeddingProvider.LOCAL or LocalEmbedder.available():
        return LocalEmbedder(settings.embedding_model or LOCAL_DEFAULT_MODEL)
    return None
