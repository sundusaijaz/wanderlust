"""Put an itinerary's stops on a map, using only coordinates from the guides.

A stop gets a position when one of the passages it cites is a listing with
coordinates (Wikivoyage listings often have them). Stops without one are
reported as not on the map rather than guessed, so every marker is backed by
the same source as the stop itself.
"""

from __future__ import annotations

import datetime as dt
from itertools import pairwise

from pydantic import BaseModel

from wanderlust.agent.itinerary import TimeOfDay
from wanderlust.agent.planner import Plan
from wanderlust.tools.distance import haversine_km


class MapStop(BaseModel):
    number: int
    """The stop's position in its day (1 = first), matching the itinerary."""
    name: str
    time_of_day: TimeOfDay
    latitude: float
    longitude: float
    source: int
    """The passage the coordinates come from."""
    indoor: bool = False


class MapDay(BaseModel):
    day: int
    title: str
    date: dt.date | None = None
    stops: list[MapStop]
    unplaced: list[str]
    """Names of stops whose passages have no coordinates."""

    @property
    def route_km(self) -> float | None:
        """Straight-line distance from stop to stop, in visiting order."""
        if len(self.stops) < 2:
            return None
        return round(
            sum(
                haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
                for a, b in pairwise(self.stops)
            ),
            2,
        )


class TripMap(BaseModel):
    days: list[MapDay]

    @property
    def placed(self) -> int:
        return sum(len(day.stops) for day in self.days)

    @property
    def total(self) -> int:
        return sum(len(day.stops) + len(day.unplaced) for day in self.days)

    @property
    def is_empty(self) -> bool:
        return self.placed == 0

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """((south, west), (north, east)) around every placed stop."""
        points = [(s.latitude, s.longitude) for day in self.days for s in day.stops]
        if not points:
            return None
        lats, lons = zip(*points, strict=True)
        return (min(lats), min(lons)), (max(lats), max(lons))

    def to_payload(self) -> dict:
        """Plain data for the web page and map files."""
        return {
            "days": [
                {**day.model_dump(mode="json"), "route_km": day.route_km} for day in self.days
            ],
            "placed": self.placed,
            "total": self.total,
            "bounds": self.bounds,
        }


def build_trip_map(plan: Plan) -> TripMap:
    days = []
    for day in plan.itinerary.days:
        placed: list[MapStop] = []
        unplaced: list[str] = []
        for number, stop in enumerate(day.stops, start=1):
            position = _position(plan, stop.sources)
            if position is None:
                unplaced.append(stop.name)
                continue
            source, latitude, longitude = position
            placed.append(
                MapStop(
                    number=number,
                    name=stop.name,
                    time_of_day=stop.time_of_day,
                    latitude=latitude,
                    longitude=longitude,
                    source=source,
                    indoor=stop.indoor,
                )
            )
        days.append(
            MapDay(day=day.day, title=day.title, date=day.date, stops=placed, unplaced=unplaced)
        )
    return TripMap(days=days)


def _position(plan: Plan, numbers: list[int]) -> tuple[int, float, float] | None:
    """The first cited passage that has coordinates."""
    for number in numbers:
        source = plan.source(number)
        if source is not None and source.chunk.lat is not None and source.chunk.lon is not None:
            return number, source.chunk.lat, source.chunk.lon
    return None
