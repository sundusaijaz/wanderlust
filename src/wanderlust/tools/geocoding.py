"""Find coordinates for place names with the Open-Meteo geocoding API."""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel

from wanderlust.tools.http import GEOCODING_URL, OpenMeteoClient, ToolError

# Place coordinates don't change; keep lookups for a month.
GEOCODING_TTL = timedelta(days=30)


class Place(BaseModel):
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    country_code: str | None = None
    region: str | None = None
    timezone: str | None = None
    population: int | None = None

    @property
    def label(self) -> str:
        """E.g. 'Sintra, Lisbon District, Portugal'."""
        parts = [self.name, self.region, self.country]
        return ", ".join(dict.fromkeys(p for p in parts if p))


class PlaceNotFoundError(ToolError):
    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__(f"No place called '{query}' was found.")


class Geocoder:
    def __init__(self, client: OpenMeteoClient, *, language: str = "en") -> None:
        self.client = client
        self.language = language

    def search(self, name: str, *, country_code: str | None = None, limit: int = 5) -> list[Place]:
        """Candidate places for a name, most relevant first (may be empty).

        ``country_code`` (ISO 3166-1 alpha-2, e.g. 'PT') narrows the results.
        """
        name = name.strip()
        if len(name) < 2:
            raise ValueError("place name must have at least 2 characters")
        params: dict[str, object] = {
            "name": name,
            "count": max(1, min(limit, 20)),
            "language": self.language,
            "format": "json",
        }
        if country_code:
            params["countryCode"] = country_code.upper()
        data = self.client.get_json(GEOCODING_URL, params, ttl=GEOCODING_TTL)
        return [_to_place(item) for item in data.get("results") or []]

    def locate(self, name: str, *, country_code: str | None = None) -> Place:
        """The best match for a name.

        Raises:
            PlaceNotFoundError: if nothing matches.
        """
        results = self.search(name, country_code=country_code, limit=1)
        if not results:
            raise PlaceNotFoundError(name)
        return results[0]


def _to_place(item: dict) -> Place:
    try:
        return Place(
            name=item["name"],
            latitude=item["latitude"],
            longitude=item["longitude"],
            country=item.get("country"),
            country_code=item.get("country_code"),
            region=item.get("admin1"),
            timezone=item.get("timezone"),
            population=item.get("population"),
        )
    except KeyError as exc:
        raise ToolError(f"Geocoding result is missing {exc}") from exc
