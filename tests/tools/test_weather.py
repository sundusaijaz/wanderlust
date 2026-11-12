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
