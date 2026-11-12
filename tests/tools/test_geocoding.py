import httpx
import pytest

from wanderlust.tools.geocoding import Geocoder, PlaceNotFoundError
from wanderlust.tools.http import ToolError

# Made-up places in the shape the Open-Meteo geocoding API returns.
RESULTS = {
    "examplestad": [
        {
            "id": 1,
            "name": "Examplestad",
            "latitude": 12.34,
            "longitude": -7.65,
            "country_code": "IM",
            "country": "Imaginaria",
            "admin1": "Coast Province",
            "timezone": "Atlantic/Example",
            "population": 12000,
        },
        {
            "id": 2,
            "name": "Examplestad",
            "latitude": 50.0,
            "longitude": 8.0,
            "country_code": "XX",
            "country": "Elsewhere",
        },
    ]
}


def geocoding_api(request):
    params = request.url.params
    matches = RESULTS.get(params["name"].casefold(), [])
    if "countryCode" in params:
        matches = [m for m in matches if m["country_code"] == params["countryCode"]]
    matches = matches[: int(params["count"])]
    return httpx.Response(200, json={"results": matches} if matches else {"generationtime_ms": 0.1})


@pytest.fixture
def geocoder(make_meteo):
    client, requests = make_meteo(geocoding_api)
    return Geocoder(client), requests


def test_search_returns_places(geocoder):
    geo, requests = geocoder
    places = geo.search("Examplestad")
    assert [p.country for p in places] == ["Imaginaria", "Elsewhere"]
    first = places[0]
    assert (first.latitude, first.longitude) == (12.34, -7.65)
    assert first.label == "Examplestad, Coast Province, Imaginaria"
    assert first.timezone == "Atlantic/Example"
    assert requests[0].url.params["language"] == "en"


def test_label_skips_missing_parts(geocoder):
    geo, _ = geocoder
    assert geo.search("Examplestad")[1].label == "Examplestad, Elsewhere"


def test_country_code_narrows_results(geocoder):
    geo, requests = geocoder
    place = geo.locate("Examplestad", country_code="xx")
    assert place.country == "Elsewhere"
    assert requests[0].url.params["countryCode"] == "XX"


def test_no_results(geocoder):
    geo, _ = geocoder
    assert geo.search("Nowhereville") == []
    with pytest.raises(PlaceNotFoundError, match="Nowhereville"):
        geo.locate("Nowhereville")


def test_lookups_are_cached(geocoder):
    geo, requests = geocoder
    geo.locate("Examplestad")
    geo.locate("Examplestad")
    assert len(requests) == 1


def test_rejects_too_short_names(geocoder):
    geo, requests = geocoder
    with pytest.raises(ValueError):
        geo.search(" x ")
    assert requests == []


def test_malformed_result(make_meteo):
    client, _ = make_meteo(lambda r: httpx.Response(200, json={"results": [{"name": "X"}]}))
    with pytest.raises(ToolError, match="missing"):
        Geocoder(client).search("Xyz")
