"""On-disk cache for parsed Wikivoyage articles."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from wanderlust.wikivoyage.models import Article

logger = logging.getLogger(__name__)

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Normalise a title so 'new york city', 'New_York_City ' share a cache entry."""
    return " ".join(title.replace("_", " ").split()).casefold()


class ArticleCache:
    """Stores articles as JSON files under ``<root>/wikivoyage/<language>/``.

    Entries older than ``ttl`` are treated as missing so guides stay fresh.
    """

    def __init__(self, root: Path, ttl: timedelta = timedelta(days=7)) -> None:
        self.root = Path(root) / "wikivoyage"
        self.ttl = ttl

    def path_for(self, language: str, title: str) -> Path:
        key = normalize_title(title)
        slug = _NON_SLUG.sub("-", key).strip("-")[:60] or "article"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        return self.root / language / f"{slug}-{digest}.json"

    def get(self, language: str, title: str, *, now: datetime | None = None) -> Article | None:
        """Return a fresh cached article, or ``None`` if missing, stale or unreadable."""
        path = self.path_for(language, title)
        try:
            article = Article.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValidationError, ValueError):
            logger.warning("Ignoring unreadable cache entry %s", path)
            return None

        if (now or datetime.now(UTC)) - article.fetched_at > self.ttl:
            return None
        return article

    def put(self, article: Article, *, requested_title: str | None = None) -> None:
        """Save an article under its canonical title and the title that was asked for."""
        titles = {article.title}
        if requested_title:
            titles.add(requested_title)
        payload = article.model_dump_json()
        for title in titles:
            self._atomic_write(self.path_for(article.language, title), payload)

    def clear(self) -> int:
        """Delete all cached articles and return how many files were removed."""
        removed = 0
        for path in self.root.glob("*/*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    @staticmethod
    def _atomic_write(path: Path, payload: str) -> None:
        # Write to a temporary file first so a crash never leaves half a JSON file.
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
