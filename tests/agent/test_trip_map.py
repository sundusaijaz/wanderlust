from datetime import date

import pytest

from wanderlust.agent.itinerary import Itinerary
from wanderlust.agent.planner import Plan, Progress
from wanderlust.agent.trip_map import build_trip_map
from wanderlust.qa.sources import Source
from wanderlust.retrieval.chunking import Chunk, ChunkKind
from wanderlust.wikivoyage.models import Attribution


def source(number, lat=None, lon=None, name=None):
    chunk = Chunk(
        id=f"chunk-{number}",
        article="Examplestad",
        section="See",
        kind=ChunkKind.LISTING if name else ChunkKind.TEXT,
        text="...",
        attribution=Attribution(title="Examplestad", url="https://en.wikivoyage.org/wiki/X"),
        listing_name=name,
        lat=lat,
        lon=lon,
    )
    return Source(number=number, chunk=chunk)


def stop(time, name, *sources, indoor=False):
    return {
        "time_of_day": time,
        "name": name,
        "description": "...",
        "sources": list(sources),
        "indoor": indoor,
    }


def make_plan(days, sources):
    itinerary = Itinerary.model_validate(
        {
            "title": "Trip",
            "summary": "...",
            "days": [
                {"day": n, "base": "Examplestad", "title": f"Day {n}", "stops": stops, **extra}
                for n, (stops, extra) in enumerate(days, start=1)
            ],
        }
    )
    return Plan(
        request="trip",
        itinerary=itinerary,
        sources=sources,
        progress=Progress(),
        provider="fake",
        model="scripted",
    )


SOURCES = [
    source(1, 38.7139, -9.1394, "Rossio"),
    source(2),  # a text passage: no coordinates
    source(3, 38.6916, -9.2160, "Belém Tower"),
    source(4, 38.6979, -9.2064, "Jerónimos"),
]


def test_stops_take_coordinates_from_their_first_located_passage():
    plan = make_plan(
        [
            (
                [
                    stop("morning", "Rossio square", 2, 1),
                    stop("afternoon", "Tram ride", 2),
                    stop("evening", "Belém", 3, 4, indoor=True),
                ],
                {"date": "2027-03-10"},
            )
        ],
        SOURCES,
    )
    trip = build_trip_map(plan)
    (day,) = trip.days
    assert day.date == date(2027, 3, 10)
    assert [(s.number, s.name, s.source) for s in day.stops] == [
        (1, "Rossio square", 1),
        (3, "Belém", 3),
    ]
    assert day.stops[1].indoor and day.stops[1].latitude == pytest.approx(38.6916)
    assert day.unplaced == ["Tram ride"]
    assert trip.placed == 2 and trip.total == 3 and not trip.is_empty
    assert day.route_km == pytest.approx(7.0, abs=0.5)  # Rossio to Belém


def test_route_needs_two_stops_and_bounds_cover_every_day():
    plan = make_plan(
        [
            ([stop("morning", "Rossio", 1)], {}),
            ([stop("morning", "Jerónimos", 4), stop("evening", "Belém Tower", 3)], {}),
        ],
        SOURCES,
    )
    trip = build_trip_map(plan)
    assert trip.days[0].route_km is None
    assert trip.days[1].route_km == pytest.approx(0.9, abs=0.2)
    assert trip.bounds == ((38.6916, -9.2160), (38.7139, -9.1394))


def test_a_plan_without_coordinates_gives_an_empty_map():
    plan = make_plan([([stop("morning", "Walk", 2)], {})], SOURCES)
    trip = build_trip_map(plan)
    assert trip.is_empty and trip.bounds is None
    assert trip.days[0].unplaced == ["Walk"]


def test_payload_is_plain_data():
    plan = make_plan([([stop("morning", "Rossio", 1), stop("evening", "Belém", 3)], {})], SOURCES)
    payload = build_trip_map(plan).to_payload()
    day = payload["days"][0]
    assert day["route_km"] == pytest.approx(7.0, abs=0.5)
    assert day["stops"][0] == {
        "number": 1,
        "name": "Rossio",
        "time_of_day": "morning",
        "latitude": 38.7139,
        "longitude": -9.1394,
        "source": 1,
        "indoor": False,
    }
    assert payload["placed"] == 2 and payload["total"] == 2
    assert payload["bounds"] == ((38.6916, -9.216), (38.7139, -9.1394))


def test_map_summaries_for_the_terminal():
    from wanderlust.plan_view import map_summary

    plan = make_plan(
        [
            ([stop("morning", "Rossio", 1), stop("evening", "Belém", 3)], {}),
            ([stop("morning", "Rossio", 1), stop("evening", "Walk", 2)], {}),
            ([stop("morning", "Walk", 2)], {}),
        ],
        SOURCES,
    )
    first, second, third = build_trip_map(plan).days
    assert map_summary(first) == "all 2 stops placed, about 7.1 km apart in a straight line"
    assert map_summary(second) == "1 of 2 stops placed"
    assert map_summary(third) == "no coordinates in the guides for these stops"
