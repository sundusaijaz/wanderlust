"""Straight-line distances and rough travel-time estimates between two points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius


class TravelMode(StrEnum):
    WALK = "walk"
    TRANSIT = "transit"
    DRIVE = "drive"


# Deliberately simple assumptions. Real routes are longer than straight lines
# (the detour factor), and each mode has a door-to-door speed plus a fixed
# overhead (waiting for a tram, finding parking). Short trips use city values,
# long trips intercity values (trains, motorways), blended in between.
@dataclass(frozen=True)
class _ModeProfile:
    city_kmh: float
    intercity_kmh: float
    city_detour: float
    intercity_detour: float
    overhead_min: float


_MODES: dict[TravelMode, _ModeProfile] = {
    TravelMode.WALK: _ModeProfile(4.5, 4.5, 1.3, 1.3, 0.0),
    TravelMode.TRANSIT: _ModeProfile(18.0, 80.0, 1.4, 1.2, 8.0),
    TravelMode.DRIVE: _ModeProfile(30.0, 90.0, 1.4, 1.2, 5.0),
}
CITY_KM = 3.0  # up to here: pure city values
INTERCITY_KM = 50.0  # from here: pure intercity values
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
    profile = _MODES[mode]
    blend = min(1.0, max(0.0, (km - CITY_KM) / (INTERCITY_KM - CITY_KM)))
    speed = profile.city_kmh + blend * (profile.intercity_kmh - profile.city_kmh)
    detour = profile.city_detour + blend * (profile.intercity_detour - profile.city_detour)
    return max(1, round(km * detour / speed * 60 + profile.overhead_min))


def distance_report(lat1: float, lon1: float, lat2: float, lon2: float) -> DistanceReport:
    km = haversine_km(lat1, lon1, lat2, lon2)
    modes = [m for m in TravelMode if m is not TravelMode.WALK or km <= MAX_WALK_KM]
    return DistanceReport(
        straight_line_km=round(km, 2),
        estimates=[TravelEstimate(mode=m, minutes=estimate_minutes(km, m)) for m in modes],
    )
