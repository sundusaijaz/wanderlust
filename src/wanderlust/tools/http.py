"""A small cached HTTP client for the Open-Meteo APIs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from wanderlust import __version__

logger = logging.getLogger(__name__)

USER_AGENT = f"Wanderlust/{__version__} (https://github.com/sundusaijaz/wanderlust)"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

Clock = Callable[[], datetime]


class ToolError(RuntimeError):
    """A travel tool could not produce a result."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JsonCache:
    """Caches JSON responses on disk, one file per request, with a lifetime per lookup."""

    def __init__(self, root: Path, *, clock: Clock = _utcnow) -> None:
        self.root = Path(root)
        self._clock = clock

    @staticmethod
    def key(url: str, params: dict[str, Any]) -> str:
        canonical = json.dumps({"url": url, "params": params}, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str, ttl: timedelta) -> Any | None:
        try:
            entry = json.loads(self._path(key).read_text(encoding="utf-8"))
            stored_at = datetime.fromisoformat(entry["stored_at"])
            data = entry["data"]
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Ignoring unreadable cache entry %s", key)
            return None
        return data if self._clock() - stored_at <= ttl else None

    def put(self, key: str, data: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"stored_at": self._clock().isoformat(), "data": data})
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class OpenMeteoClient:
    """GETs JSON from Open-Meteo, reusing cached responses while they are fresh.

    Open-Meteo is free for non-commercial use; caching keeps Wanderlust polite
    (repeated questions about the same city don't re-download anything).
    """

    def __init__(
        self,
        *,
        cache: JsonCache | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 15.0,
        clock: Clock = _utcnow,
    ) -> None:
        self.cache = cache
        self.clock = clock
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            transport=httpx.HTTPTransport(retries=2),
        )

    def get_json(self, url: str, params: dict[str, Any], *, ttl: timedelta) -> dict[str, Any]:
        key = JsonCache.key(url, params)
        if self.cache is not None:
            cached = self.cache.get(key, ttl)
            if cached is not None:
                return cached

        try:
            response = self._http.get(url, params=params)
        except httpx.HTTPError as exc:
            raise ToolError(f"Could not reach Open-Meteo: {exc}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise ToolError(
                f"Open-Meteo returned invalid JSON (HTTP {response.status_code})"
            ) from exc

        if isinstance(data, dict) and data.get("error"):
            raise ToolError(f"Open-Meteo: {data.get('reason', 'unknown error')}")
        if response.is_error:
            raise ToolError(f"Open-Meteo returned HTTP {response.status_code}")
        if not isinstance(data, dict):
            raise ToolError("Open-Meteo returned an unexpected response")

        if self.cache is not None:
            self.cache.put(key, data)
        return data

    def close(self) -> None:
        if self._owns_client:
            self._http.close()
