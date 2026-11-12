"""Weather forecasts from Open-Meteo."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel

from wanderlust.tools.http import FORECAST_URL, OpenMeteoClient, ToolError

ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
FORECAST_TTL = timedelta(hours=1)
FORECAST_HORIZON_DAYS = 16  # today plus the next 15 days
MAX_RANGE_DAYS = 16

DAILY_FIELDS = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
)

# WMO weather interpretation codes, as used by Open-Meteo.
WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Freezing fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Light freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Violent showers",
    85: "Light snow showers",
    86: "Snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}
_WET_CODES = frozenset(code for code in WMO_CODES if code >= 51)


def describe_weather(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WMO_CODES.get(code, f"Weather code {code}")


class DayWeather(BaseModel):
    date: date
    weather_code: int | None = None
    temp_max_c: float | None = None
    temp_min_c: float | None = None
    precipitation_mm: float | None = None
    precipitation_probability: int | None = None

    @property
    def description(self) -> str:
        return describe_weather(self.weather_code)

    @property
    def is_wet(self) -> bool:
        """Whether to plan indoor alternatives for this day."""
        if self.precipitation_probability is not None and self.precipitation_probability >= 60:
            return True
        if self.precipitation_mm is not None and self.precipitation_mm >= 2.0:
            return True
        return self.weather_code in _WET_CODES and (self.precipitation_mm or 0) >= 0.5


class WeatherReport(BaseModel):
    latitude: float
    longitude: float
    kind: Literal["forecast", "typical"]
    days: list[DayWeather]
    note: str | None = None
    attribution: str = ATTRIBUTION


def check_range(start: date, end: date) -> None:
    if end < start:
        raise ValueError("end date must not be before start date")
    if (end - start).days + 1 > MAX_RANGE_DAYS:
        raise ValueError(f"date range must be at most {MAX_RANGE_DAYS} days")


def parse_daily(data: dict[str, Any]) -> list[DayWeather]:
    """Turn Open-Meteo's column-wise 'daily' block into one object per day."""
    try:
        daily = data["daily"]
        times = daily["time"]
    except (KeyError, TypeError) as exc:
        raise ToolError("Open-Meteo response has no daily data") from exc

    def column(name: str) -> list[Any]:
        values = daily.get(name) or []
        return list(values) + [None] * (len(times) - len(values))

    codes = column("weather_code")
    highs, lows = column("temperature_2m_max"), column("temperature_2m_min")
    rain, chance = column("precipitation_sum"), column("precipitation_probability_max")
    return [
        DayWeather(
            date=date.fromisoformat(day),
            weather_code=codes[i],
            temp_max_c=highs[i],
            temp_min_c=lows[i],
            precipitation_mm=rain[i],
            precipitation_probability=chance[i],
        )
        for i, day in enumerate(times)
    ]


class WeatherService:
    def __init__(self, client: OpenMeteoClient) -> None:
        self.client = client

    def today(self) -> date:
        return self.client.clock().date()

    def forecast(self, latitude: float, longitude: float, start: date, end: date) -> WeatherReport:
        """Daily forecast for dates from today up to 15 days ahead."""
        check_range(start, end)
        today = self.today()
        last = today + timedelta(days=FORECAST_HORIZON_DAYS - 1)
        if start < today or end > last:
            raise ValueError(
                f"forecasts cover {today.isoformat()} to {last.isoformat()}; "
                "use typical weather for other dates"
            )
        data = self.client.get_json(
            FORECAST_URL,
            {
                "latitude": round(latitude, 3),
                "longitude": round(longitude, 3),
                "daily": ",".join(DAILY_FIELDS),
                "timezone": "auto",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
            ttl=FORECAST_TTL,
        )
        return WeatherReport(
            latitude=latitude, longitude=longitude, kind="forecast", days=parse_daily(data)
        )
