"""Searchable guide indexes and their on-disk storage."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from wanderlust.retrieval.chunking import Chunk
from wanderlust.retrieval.keyword import BM25Index
from wanderlust.wikivoyage.cache import title_slug

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
_CHUNKS = TypeAdapter(list[Chunk])


class IndexLoadError(RuntimeError):
    """A saved index is missing, corrupt or from an incompatible version."""


class IndexMeta(BaseModel):
    """What an index contains and how it was built."""

    format_version: int = FORMAT_VERSION
    destination: str
    language: str
    articles: dict[str, int] = Field(description="Article title -> revision id")
    embedder: str | None = None
    dims: int | None = None
    chunk_count: int
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VectorIndex:
    """Exact cosine-similarity search over unit-length vectors.

    A destination produces a few hundred chunks, so a brute-force matrix
    product is fast and needs no extra infrastructure.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        self.vectors = vectors

    @property
    def dims(self) -> int:
        return self.vectors.shape[1]

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dims:
            raise ValueError(f"query has {query.shape[0]} dimensions, index has {self.dims}")
        if not len(self) or k <= 0:
            return []
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        scores = self.vectors @ (query / norm)
        k = min(k, len(self))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


@dataclass
class GuideIndex:
    """Chunks of one destination plus the keyword and (optional) vector indexes."""

    meta: IndexMeta
    chunks: list[Chunk]
    vectors: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.chunks) != self.meta.chunk_count:
            raise ValueError("meta.chunk_count does not match the number of chunks")
        if self.vectors is not None and self.vectors.shape != (len(self.chunks), self.meta.dims):
            raise ValueError("vectors do not match the chunks and meta.dims")

    @cached_property
    def bm25(self) -> BM25Index:
        return BM25Index([chunk.search_text for chunk in self.chunks])

    @cached_property
    def vector_index(self) -> VectorIndex | None:
        return VectorIndex(self.vectors) if self.vectors is not None else None

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(directory / "chunks.json", _CHUNKS.dump_json(self.chunks))
        vectors_path = directory / "vectors.npy"
        if self.vectors is not None:
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".npy")
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, self.vectors.astype(np.float32))
            os.replace(tmp, vectors_path)
        else:
            vectors_path.unlink(missing_ok=True)
        # Written last: a readable meta.json means the other files are complete.
        _atomic_write(directory / "meta.json", self.meta.model_dump_json(indent=2).encode())

    @classmethod
    def load(cls, directory: Path) -> GuideIndex:
        try:
            raw_meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if raw_meta.get("format_version") != FORMAT_VERSION:
                raise IndexLoadError(
                    f"Index format {raw_meta.get('format_version')} is not supported; rebuild it."
                )
            meta = IndexMeta.model_validate(raw_meta)
            chunks = _CHUNKS.validate_json((directory / "chunks.json").read_bytes())
            vectors = None
            if meta.embedder is not None:
                vectors = np.load(directory / "vectors.npy", allow_pickle=False)
            return cls(meta=meta, chunks=chunks, vectors=vectors)
        except IndexLoadError:
            raise
        except FileNotFoundError as exc:
            raise IndexLoadError(f"Index files missing in {directory}") from exc
        except (OSError, ValueError, ValidationError) as exc:
            raise IndexLoadError(f"Index in {directory} is unreadable: {exc}") from exc


class IndexStore:
    """Keeps one index per destination under ``<data_dir>/index/<language>/``."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "index"

    def path_for(self, language: str, destination: str) -> Path:
        return self.root / language / title_slug(destination)

    def save(self, index: GuideIndex) -> Path:
        path = self.path_for(index.meta.language, index.meta.destination)
        index.save(path)
        return path

    def load(self, language: str, destination: str) -> GuideIndex | None:
        """The saved index for a destination, or ``None`` if there is no usable one."""
        path = self.path_for(language, destination)
        if not (path / "meta.json").exists():
            return None
        try:
            return GuideIndex.load(path)
        except IndexLoadError as exc:
            logger.warning("Ignoring saved index: %s", exc)
            return None


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
