import pytest

from wanderlust.tools.distance import (
    TravelMode,
    distance_report,
    estimate_minutes,
    haversine_km,
)


def test_haversine_known_distances():
    assert haversine_km(0, 0, 0, 0) == 0
    # One degree of longitude on the equator is about 111.2 km.
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.05)
    # Lisbon (Rossio) to Belém Tower: about 7 km in a straight line.
    assert haversine_km(38.7139, -9.1394, 38.6916, -9.2160) == pytest.approx(7.0, abs=0.5)


def test_haversine_is_symmetric():
    assert haversine_km(38.7, -9.1, 41.1, -8.6) == pytest.approx(
        haversine_km(41.1, -8.6, 38.7, -9.1)
    )


@pytest.mark.parametrize("coords", [(91, 0, 0, 0), (0, 0, 0, 181), (0, -181, 0, 0)])
def test_haversine_rejects_invalid_coordinates(coords):
    with pytest.raises(ValueError):
        haversine_km(*coords)


def test_estimates_grow_with_distance_and_include_overheads():
    assert estimate_minutes(1.0, TravelMode.WALK) == 17  # 1.3 km at 4.5 km/h
    assert estimate_minutes(0.0, TravelMode.WALK) == 1  # never zero
    assert estimate_minutes(5, TravelMode.DRIVE) > estimate_minutes(2, TravelMode.DRIVE)


def test_short_trips_include_walking():
    report = distance_report(38.7139, -9.1394, 38.7110, -9.1330)
    assert report.straight_line_km < 1
    assert report.walkable
    assert [e.mode for e in report.estimates] == [
        TravelMode.WALK,
        TravelMode.TRANSIT,
        TravelMode.DRIVE,
    ]
    assert "Rough estimates" in report.note


def test_long_trips_drop_walking():
    lisbon_to_porto = distance_report(38.7223, -9.1393, 41.1579, -8.6291)
    assert lisbon_to_porto.straight_line_km == pytest.approx(274, abs=3)
    assert not lisbon_to_porto.walkable


@pytest.mark.parametrize(
    ("km", "mode", "low", "high"),
    [
        # Rossio to Belém by tram (~7 km): about half an hour.
        (7.0, TravelMode.TRANSIT, 25, 40),
        # Lisbon to Sintra (~21 km): train about 40-50 min door to door, car about 35 min.
        (21.5, TravelMode.TRANSIT, 40, 60),
        (21.5, TravelMode.DRIVE, 30, 45),
        # Lisbon to Porto (~274 km): about 3-4.5 hours by train or car.
        (274, TravelMode.TRANSIT, 180, 270),
        (274, TravelMode.DRIVE, 180, 240),
    ],
)
def test_estimates_are_plausible_for_known_trips(km, mode, low, high):
    assert low <= estimate_minutes(km, mode) <= high


def test_estimates_never_decrease_with_distance():
    for mode in TravelMode:
        times = [estimate_minutes(km / 2, mode) for km in range(0, 400)]
        assert times == sorted(times)
