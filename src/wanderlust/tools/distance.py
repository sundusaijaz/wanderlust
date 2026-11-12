"""Straight-line distances and rough travel-time estimates between two points."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, Field

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius


class TravelMode(StrEnum):
    WALK = "walk"
    TRANSIT = "transit"
    DRIVE = "drive"


# Deliberately simple assumptions: real routes are longer than straight lines
# ("detour"), and each mode has an average door-to-door speed plus a fixed
# overhead (waiting for a tram, finding parking).
_MODES: dict[TravelMode, tuple[float, float, float]] = {
    #                  speed km/h, detour factor, overhead minutes
    TravelMode.WALK: (4.5, 1.3, 0.0),
    TravelMode.TRANSIT: (18.0, 1.4, 8.0),
    TravelMode.DRIVE: (30.0, 1.4, 5.0),
}
MAX_WALK_KM = 6.0  # beyond this, walking is not suggested


class TravelEstimate(BaseModel):
    mode: TravelMode
    minutes: int


class DistanceReport(BaseModel):
    straight_line_km: float
    estimates: list[TravelEstimate] = Field(default_factory=list)
    note: str = (
        "Rough estimates from straight-line distance; real routes, hills and "
        "timetables can make trips longer."
    )

    @property
    def walkable(self) -> bool:
        return any(e.mode is TravelMode.WALK for e in self.estimates)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates, in kilometres."""
    for lat in (lat1, lat2):
        if not -90 <= lat <= 90:
            raise ValueError("latitude must be between -90 and 90")
    for lon in (lon1, lon2):
        if not -180 <= lon <= 180:
            raise ValueError("longitude must be between -180 and 180")
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def estimate_minutes(km: float, mode: TravelMode) -> int:
    speed, detour, overhead = _MODES[mode]
    return max(1, round(km * detour / speed * 60 + overhead))


def distance_report(lat1: float, lon1: float, lat2: float, lon2: float) -> DistanceReport:
    km = haversine_km(lat1, lon1, lat2, lon2)
    modes = [m for m in TravelMode if m is not TravelMode.WALK or km <= MAX_WALK_KM]
    return DistanceReport(
        straight_line_km=round(km, 2),
        estimates=[TravelEstimate(mode=m, minutes=estimate_minutes(km, m)) for m in modes],
    )
