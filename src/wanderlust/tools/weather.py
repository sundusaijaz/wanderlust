"""Weather forecasts from Open-Meteo."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel

from wanderlust.tools.http import ARCHIVE_URL, FORECAST_URL, OpenMeteoClient, ToolError

ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
FORECAST_TTL = timedelta(hours=1)
ARCHIVE_TTL = timedelta(days=30)  # past weather doesn't change
TYPICAL_YEARS = 3
ARCHIVE_DELAY_DAYS = 7
WET_DAY_MM = 1.0
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

    def typical(
        self,
        latitude: float,
        longitude: float,
        start: date,
        end: date,
        *,
        years: int = TYPICAL_YEARS,
    ) -> WeatherReport:
        """What the weather is usually like on these dates, from the previous years' records.

        For each trip day, temperatures and rainfall are averaged over the same
        calendar day in each of the last ``years`` years; the rain chance is the
        share of those years with at least 1 mm of rain.
        """
        check_range(start, end)
        if years < 1:
            raise ValueError("years must be at least 1")
        trip_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

        # The most recent year whose matching dates are already in the archive,
        # which lags real time by a few days.
        archived_until = self.today() - timedelta(days=ARCHIVE_DELAY_DAYS)
        latest = start.year - 1
        while _same_day_in(latest + (end.year - start.year), end) > archived_until:
            latest -= 1
        past_years = [latest - i for i in range(years)]

        samples: dict[date, list[DayWeather]] = {day: [] for day in trip_days}
        for year in past_years:
            shifted = {day: _same_day_in(year + (day.year - start.year), day) for day in trip_days}
            data = self.client.get_json(
                ARCHIVE_URL,
                {
                    "latitude": round(latitude, 3),
                    "longitude": round(longitude, 3),
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "timezone": "auto",
                    "start_date": min(shifted.values()).isoformat(),
                    "end_date": max(shifted.values()).isoformat(),
                },
                ttl=ARCHIVE_TTL,
            )
            by_date = {d.date: d for d in parse_daily(data)}
            for day, past_day in shifted.items():
                if past_day in by_date:
                    samples[day].append(by_date[past_day])

        days = [_summarise(day, samples[day]) for day in trip_days]
        span = f"{min(past_years)}-{max(past_years)}" if len(past_years) > 1 else str(past_years[0])
        return WeatherReport(
            latitude=latitude,
            longitude=longitude,
            kind="typical",
            days=days,
            note=f"Typical weather from the same dates in {span}; not a forecast.",
        )

    def for_dates(self, latitude: float, longitude: float, start: date, end: date) -> WeatherReport:
        """A forecast when the dates are close enough, otherwise typical weather."""
        check_range(start, end)
        today = self.today()
        if start < today:
            raise ValueError("dates must not be in the past")
        if end <= today + timedelta(days=FORECAST_HORIZON_DAYS - 1):
            return self.forecast(latitude, longitude, start, end)
        return self.typical(latitude, longitude, start, end)


def _same_day_in(year: int, day: date) -> date:
    """The same calendar day in another year (29 February becomes the 28th)."""
    try:
        return day.replace(year=year)
    except ValueError:
        return day.replace(year=year, day=28)


def _summarise(day: date, samples: list[DayWeather]) -> DayWeather:
    def avg(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return round(mean(present), 1) if present else None

    rain = [s.precipitation_mm for s in samples if s.precipitation_mm is not None]
    codes = [s.weather_code for s in samples if s.weather_code is not None]
    return DayWeather(
        date=day,
        weather_code=Counter(codes).most_common(1)[0][0] if codes else None,
        temp_max_c=avg([s.temp_max_c for s in samples]),
        temp_min_c=avg([s.temp_min_c for s in samples]),
        precipitation_mm=avg(rain),
        precipitation_probability=(
            round(100 * sum(r >= WET_DAY_MM for r in rain) / len(rain)) if rain else None
        ),
    )
