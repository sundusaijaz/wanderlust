"""Small client for the Wikivoyage MediaWiki API."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import httpx

from wanderlust import __version__
from wanderlust.wikivoyage.models import Article
from wanderlust.wikivoyage.parser import parse_article

# Wikimedia asks API clients to identify themselves with a descriptive User-Agent.
USER_AGENT = f"Wanderlust/{__version__} (https://github.com/sundusaijaz/wanderlust)"


class WikivoyageError(RuntimeError):
    """Wikivoyage could not be reached or returned an unexpected response."""


class ArticleNotFoundError(WikivoyageError):
    """No article exists with the requested title."""

    def __init__(self, title: str, suggestion: str | None = None) -> None:
        self.title = title
        self.suggestion = suggestion
        message = f"No Wikivoyage article found for '{title}'."
        if suggestion:
            message += f" Did you mean '{suggestion}'?"
        super().__init__(message)


class WikivoyageClient:
    """Fetch and parse Wikivoyage articles.

    Pass your own ``http_client`` to control transport, retries or (in tests)
    to mock responses.
    """

    def __init__(
        self,
        language: str = "en",
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.language = language
        self.endpoint = f"https://{language}.wikivoyage.org/w/api.php"
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            transport=httpx.HTTPTransport(retries=2),
            follow_redirects=True,
        )

    def fetch_article(self, title: str) -> Article:
        """Download and parse the article called ``title``, following redirects.

        Raises:
            ArticleNotFoundError: if there is no such article.
            WikivoyageError: on network errors or unexpected responses.
        """
        title = title.strip()
        if not title:
            raise ValueError("title must not be empty")

        data = self._get(
            action="parse",
            page=title,
            prop="wikitext|revid",
            redirects=1,
        )
        error = data.get("error")
        if error:
            if error.get("code") == "missingtitle":
                raise ArticleNotFoundError(title, suggestion=self.suggest(title))
            raise WikivoyageError(f"Wikivoyage API error: {error.get('info', error)}")

        try:
            page = data["parse"]
            return parse_article(
                page["wikitext"],
                title=page["title"],
                language=self.language,
                page_id=int(page["pageid"]),
                revision_id=int(page["revid"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WikivoyageError("Unexpected response format from Wikivoyage") from exc

    def suggest(self, query: str) -> str | None:
        """Best-guess article title for a misspelled or partial query."""
        try:
            data = self._get(action="query", list="search", srsearch=query, srlimit=1)
        except WikivoyageError:
            return None
        search = data.get("query", {})
        hits = search.get("search") or []
        if hits:
            return hits[0].get("title")
        suggestion = search.get("searchinfo", {}).get("suggestion")
        return suggestion.title() if suggestion else None

    def _get(self, **params: Any) -> dict[str, Any]:
        params |= {"format": "json", "formatversion": 2}
        try:
            response = self._http.get(self.endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise WikivoyageError(f"Wikivoyage returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise WikivoyageError(f"Could not reach Wikivoyage: {exc}") from exc
        except ValueError as exc:
            raise WikivoyageError("Wikivoyage returned invalid JSON") from exc

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
