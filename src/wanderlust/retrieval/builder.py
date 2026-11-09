"""Build a searchable index for a destination."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from wanderlust.retrieval.chunking import DEFAULT_MAX_CHARS, chunk_article
from wanderlust.retrieval.embeddings import Embedder
from wanderlust.retrieval.index import GuideIndex, IndexMeta
from wanderlust.wikivoyage.models import Article
from wanderlust.wikivoyage.service import GuideLoader

ProgressCallback = Callable[[str], None]


@dataclass
class BuildReport:
    index: GuideIndex
    articles: list[Article]
    skipped: list[str] = field(default_factory=list)


def build_index(
    destination: str,
    loader: GuideLoader,
    embedder: Embedder | None,
    *,
    refresh: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    on_progress: ProgressCallback | None = None,
) -> BuildReport:
    """Load a destination's guide, chunk it and embed the chunks (if an embedder is given)."""
    progress = on_progress or (lambda message: None)

    progress(f"Loading the guide for {destination}")
    main = loader.load(destination, refresh=refresh).article
    articles = [main]

    chunks = [
        chunk for article in articles for chunk in chunk_article(article, max_chars=max_chars)
    ]
    if not chunks:
        raise ValueError(f"The guide for {main.title} has no text to index.")

    vectors = None
    if embedder is not None:
        progress(f"Embedding {len(chunks)} passages with {embedder.name}")
        vectors = embedder.embed([chunk.search_text for chunk in chunks])

    meta = IndexMeta(
        destination=main.title,
        language=main.language,
        articles={article.title: article.revision_id for article in articles},
        embedder=embedder.name if embedder is not None else None,
        dims=int(vectors.shape[1]) if vectors is not None else None,
        chunk_count=len(chunks),
    )
    return BuildReport(
        index=GuideIndex(meta=meta, chunks=chunks, vectors=vectors), articles=articles
    )
