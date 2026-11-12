from datetime import date

import httpx
import pytest

from wanderlust.tools.http import ToolError
from wanderlust.tools.weather import DayWeather, WeatherService, describe_weather, parse_daily


def forecast_api(request):
    params = request.url.params
    start, end = date.fromisoformat(params["start_date"]), date.fromisoformat(params["end_date"])
    days = [date.fromordinal(d).isoformat() for d in range(start.toordinal(), end.toordinal() + 1)]
    n = len(days)
    return httpx.Response(
        200,
        json={
            "daily": {
                "time": days,
                "weather_code": ([0, 63, 80] * n)[:n],
                "temperature_2m_max": ([21.5, 17.0, 18.2] * n)[:n],
                "temperature_2m_min": ([14.0, 12.5, 13.1] * n)[:n],
                "precipitation_sum": ([0.0, 8.4, 0.6] * n)[:n],
                "precipitation_probability_max": ([5, 90, 40] * n)[:n],
            }
        },
    )


@pytest.fixture
def weather(make_meteo):
    client, requests = make_meteo(forecast_api)
    return WeatherService(client), requests


def test_describe_weather():
    assert describe_weather(0) == "Clear sky"
    assert describe_weather(95) == "Thunderstorm"
    assert describe_weather(42) == "Weather code 42"
    assert describe_weather(None) == "Unknown"


def test_forecast_parses_days(weather):
    service, requests = weather
    report = service.forecast(38.7223, -9.1393, date(2026, 11, 13), date(2026, 11, 15))

    assert report.kind == "forecast"
    assert [d.date for d in report.days] == [
        date(2026, 11, 13),
        date(2026, 11, 14),
        date(2026, 11, 15),
    ]
    sunny, rainy, showers = report.days
    assert (sunny.description, sunny.temp_max_c, sunny.is_wet) == ("Clear sky", 21.5, False)
    assert (rainy.description, rainy.precipitation_probability, rainy.is_wet) == ("Rain", 90, True)
    assert showers.is_wet  # showers with measurable rain
    assert "Open-Meteo" in report.attribution

    params = requests[0].url.params
    assert params["latitude"] == "38.722"
    assert params["timezone"] == "auto"
    assert "precipitation_probability_max" in params["daily"]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (date(2026, 11, 15), date(2026, 11, 13), "before start"),
        (date(2026, 11, 10), date(2026, 11, 13), "forecasts cover"),
        (date(2026, 11, 20), date(2026, 11, 30), "forecasts cover"),
        (date(2026, 11, 12), date(2026, 12, 12), "at most 16 days"),
    ],
)
def test_forecast_date_validation(weather, start, end, message):
    service, requests = weather
    with pytest.raises(ValueError, match=message):
        service.forecast(0, 0, start, end)
    assert requests == []


def test_parse_daily_pads_missing_columns():
    days = parse_daily({"daily": {"time": ["2026-03-01", "2026-03-02"], "weather_code": [1]}})
    assert days[1].weather_code is None
    assert days[1].temp_max_c is None


def test_parse_daily_without_daily_block():
    with pytest.raises(ToolError, match="no daily data"):
        parse_daily({"hourly": {}})


@pytest.mark.parametrize(
    ("day", "wet"),
    [
        (DayWeather(date=date(2026, 1, 1), precipitation_probability=70), True),
        (DayWeather(date=date(2026, 1, 1), precipitation_mm=3.0), True),
        (DayWeather(date=date(2026, 1, 1), weather_code=51, precipitation_mm=0.1), False),
        (DayWeather(date=date(2026, 1, 1), weather_code=2, precipitation_mm=0.0), False),
    ],
)
def test_is_wet(day, wet):
    assert day.is_wet is wet


def archive_api(request):
    """Past weather: wet in odd years, dry in even years, warmer each year."""
    params = request.url.params
    start, end = date.fromisoformat(params["start_date"]), date.fromisoformat(params["end_date"])
    days = [date.fromordinal(d) for d in range(start.toordinal(), end.toordinal() + 1)]
    wet = start.year % 2 == 1
    return httpx.Response(
        200,
        json={
            "daily": {
                "time": [d.isoformat() for d in days],
                "weather_code": [63 if wet else 1 for _ in days],
                "temperature_2m_max": [15.0 + (start.year - 2023) for _ in days],
                "temperature_2m_min": [9.0 for _ in days],
                "precipitation_sum": [6.0 if wet else 0.0 for _ in days],
            }
        },
    )


def test_typical_weather_averages_previous_years(make_meteo):
    client, requests = make_meteo(archive_api)
    report = WeatherService(client).typical(38.72, -9.14, date(2027, 3, 1), date(2027, 3, 2))

    assert report.kind == "typical"
    assert "2024-2026" in report.note and "not a forecast" in report.note
    assert sorted(r.url.params["start_date"] for r in requests) == [
        "2024-03-01",
        "2025-03-01",
        "2026-03-01",
    ]
    day = report.days[0]
    assert day.date == date(2027, 3, 1)
    assert day.temp_max_c == 17.0  # mean of 16, 17, 18
    assert day.temp_min_c == 9.0
    assert day.precipitation_probability == 33  # wet in 1 of 3 years (2025)
    assert day.precipitation_mm == 2.0
    assert day.weather_code == 1  # dry was more common


def test_typical_weather_handles_leap_day(make_meteo):
    client, requests = make_meteo(archive_api)
    report = WeatherService(client).typical(0, 0, date(2028, 2, 29), date(2028, 2, 29), years=1)
    assert report.days[0].date == date(2028, 2, 29)
    assert requests[0].url.params["start_date"] == "2026-02-28"


def test_typical_weather_is_cached(make_meteo):
    client, requests = make_meteo(archive_api)
    service = WeatherService(client)
    service.typical(0, 0, date(2027, 3, 1), date(2027, 3, 3))
    service.typical(0, 0, date(2027, 3, 1), date(2027, 3, 3))
    assert len(requests) == 3


def test_for_dates_chooses_forecast_or_typical(make_meteo):
    def api(request):
        return (forecast_api if "forecast" in str(request.url) else archive_api)(request)

    client, _ = make_meteo(api)
    service = WeatherService(client)
    assert service.for_dates(0, 0, date(2026, 11, 13), date(2026, 11, 14)).kind == "forecast"
    assert service.for_dates(0, 0, date(2027, 3, 1), date(2027, 3, 4)).kind == "typical"
    with pytest.raises(ValueError, match="past"):
        service.for_dates(0, 0, date(2026, 11, 1), date(2026, 11, 2))


def test_typical_rejects_bad_years(make_meteo):
    client, _ = make_meteo(archive_api)
    with pytest.raises(ValueError):
        WeatherService(client).typical(0, 0, date(2027, 3, 1), date(2027, 3, 1), years=0)


def test_typical_weather_skips_years_not_yet_archived(make_meteo):
    client, requests = make_meteo(archive_api)
    # Today is 2026-11-12 in tests: last year's 20 November is not in the archive yet.
    WeatherService(client).typical(0, 0, date(2027, 11, 20), date(2027, 11, 21), years=2)
    assert sorted(r.url.params["start_date"] for r in requests) == ["2024-11-20", "2025-11-20"]
