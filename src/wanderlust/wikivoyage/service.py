"""Load guides through the cache, falling back to the Wikivoyage API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from wanderlust.config import Settings
from wanderlust.wikivoyage.cache import ArticleCache
from wanderlust.wikivoyage.client import WikivoyageClient
from wanderlust.wikivoyage.models import Article


@dataclass(frozen=True)
class LoadedArticle:
    article: Article
    from_cache: bool


class GuideLoader:
    """Cache-first access to Wikivoyage articles."""

    def __init__(self, client: WikivoyageClient, cache: ArticleCache) -> None:
        self.client = client
        self.cache = cache

    def load(self, title: str, *, refresh: bool = False) -> LoadedArticle:
        if not refresh:
            cached = self.cache.get(self.client.language, title)
            if cached is not None:
                return LoadedArticle(cached, from_cache=True)

        article = self.client.fetch_article(title)
        self.cache.put(article, requested_title=title)
        return LoadedArticle(article, from_cache=False)

    def close(self) -> None:
        self.client.close()


def build_loader(settings: Settings) -> GuideLoader:
    """Create a loader configured from application settings."""
    return GuideLoader(
        client=WikivoyageClient(settings.wikivoyage_language),
        cache=ArticleCache(
            settings.data_dir, ttl=timedelta(hours=settings.wikivoyage_cache_ttl_hours)
        ),
    )
