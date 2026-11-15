import json
from datetime import date

import pytest
from pydantic import ValidationError

from wanderlust.agent.itinerary import Itinerary, TimeOfDay
from wanderlust.tools.registry import json_schema


def stop(time="morning", name="Old Lighthouse", sources=(1,), **extra):
    return {
        "time_of_day": time,
        "name": name,
        "description": "Climb the steps.",
        "sources": list(sources),
        **extra,
    }


def day(n, *stops, **extra):
    return {
        "day": n,
        "base": "Examplestad",
        "title": f"Day {n}",
        "stops": list(stops) or [stop()],
        **extra,
    }


def plan(*days, **extra):
    return {"title": "A weekend", "summary": "Sea air.", "days": list(days), **extra}


def test_valid_itinerary():
    itinerary = Itinerary.model_validate(
        plan(
            day(1, stop("morning"), stop("evening", "Harbour Kitchen", (2, 1)), date="2027-03-10"),
            day(2, date="2027-03-11"),
            tips=["Take the tram [3]."],
        )
    )
    assert itinerary.start_date == date(2027, 3, 10)
    assert itinerary.days[0].stops[1].time_of_day is TimeOfDay.EVENING
    assert itinerary.cited_numbers() == [1, 2, 3]
    assert [s.name for _, s in itinerary.stops()] == [
        "Old Lighthouse",
        "Harbour Kitchen",
        "Old Lighthouse",
    ]


def test_dates_are_optional():
    itinerary = Itinerary.model_validate(plan(day(1), day(2)))
    assert itinerary.start_date is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (plan(day(1), day(3)), "numbered 1, 2, 3"),
        (plan(day(2)), "numbered 1, 2, 3"),
        (plan(day(1, date="2027-03-10"), day(2)), "every day or for none"),
        (plan(day(1, date="2027-03-10"), day(2, date="2027-03-12")), "consecutive"),
        (plan(day(1, stop("evening"), stop("morning"))), "ordered morning, afternoon, evening"),
        (plan(day(1, stop(sources=()))), "sources"),
        (plan(day(1, stop(time="midnight"))), "time_of_day"),
        (plan(day(1, stop(price="cheap"))), "Extra inputs"),
        (plan(), "days"),
        (plan(day(1), title=""), "title"),
    ],
)
def test_invalid_itineraries(data, message):
    with pytest.raises(ValidationError, match=message):
        Itinerary.model_validate(data)


def test_schema_is_self_contained_for_tool_use():
    schema = json_schema(Itinerary)
    text = json.dumps(schema)
    assert "$defs" not in text and "$ref" not in text
    stops = schema["properties"]["days"]["items"]["properties"]["stops"]["items"]
    assert stops["properties"]["time_of_day"]["enum"] == ["morning", "afternoon", "evening"]
    assert stops["additionalProperties"] is False
    assert set(schema["required"]) == {"title", "summary", "days"}
