"""Expose the travel tools to a language model and run the calls it makes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wanderlust.llm.types import ToolCall, ToolResult, ToolSpec
from wanderlust.tools.distance import distance_report
from wanderlust.tools.geocoding import Geocoder
from wanderlust.tools.http import JsonCache, OpenMeteoClient, ToolError
from wanderlust.tools.weather import WeatherReport, WeatherService


@dataclass(frozen=True)
class Tool:
    """A tool the model can call: typed arguments plus a handler."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], Any]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description, parameters=json_schema(self.args_model)
        )


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON Schema, self-contained and without Pydantic's 'title' noise.

    Nested models are inlined instead of referenced through ``$defs``, because
    tool schemas are easier for every provider (and model) to read that way.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any, seen: tuple[str, ...] = ()) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name in seen:
                    raise ValueError(f"recursive model '{name}' cannot be inlined")
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**defs[name], **extra}, (*seen, name))
            return {k: resolve(v, seen) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        return node

    return resolve(schema)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name '{tool.name}'")
            self._tools[tool.name] = tool

    def __add__(self, other: ToolRegistry) -> ToolRegistry:
        """A registry with the tools of both (names must not clash)."""
        return ToolRegistry([*self.tools, *other.tools])

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def run(self, call: ToolCall) -> ToolResult:
        """Execute one call. Problems become error results the model can react to."""
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(call, f"Unknown tool '{call.name}'. Available: {', '.join(self.names)}.")
        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return _error(call, f"Invalid arguments for {call.name}: {validation_problems(exc)}")
        try:
            result = tool.handler(args)
        except (ToolError, ValueError) as exc:
            return _error(call, str(exc))
        return ToolResult(call_id=call.id, content=_to_text(result))


def validation_problems(exc: ValidationError) -> str:
    """One line listing each problem with its location, e.g. 'days.0.stops: ...'."""
    return "; ".join(
        f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()
    )


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, content=message, is_error=True)


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json", exclude_none=True)
    return json.dumps(result, ensure_ascii=False, default=str)


# -- argument models ----------------------------------------------------------


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeocodeArgs(_Args):
    place: str = Field(min_length=2, description="Place name, e.g. 'Sintra' or 'Belém, Lisbon'.")
    country_code: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z]{2}$",
        description="Optional two-letter country code to disambiguate, e.g. 'PT'.",
    )


class WeatherArgs(_Args):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    start_date: date = Field(description="First day of the trip (YYYY-MM-DD).")
    end_date: date = Field(description="Last day of the trip (YYYY-MM-DD), at most 16 days later.")


class DistanceArgs(_Args):
    from_latitude: float = Field(ge=-90, le=90)
    from_longitude: float = Field(ge=-180, le=180)
    to_latitude: float = Field(ge=-90, le=90)
    to_longitude: float = Field(ge=-180, le=180)


# -- the travel tools ---------------------------------------------------------


def weather_payload(report: WeatherReport) -> dict[str, Any]:
    """A compact, model-friendly view of a weather report."""
    payload: dict[str, Any] = {
        "kind": report.kind,
        "days": [
            {
                "date": day.date.isoformat(),
                "summary": day.description,
                "high_c": day.temp_max_c,
                "low_c": day.temp_min_c,
                "rain_mm": day.precipitation_mm,
                "rain_chance_percent": day.precipitation_probability,
                "plan_indoor_alternatives": day.is_wet,
            }
            for day in report.days
        ],
        "attribution": report.attribution,
    }
    if report.note:
        payload["note"] = report.note
    return payload


def travel_tools(client: OpenMeteoClient, *, language: str = "en") -> ToolRegistry:
    geocoder = Geocoder(client, language=language)
    weather = WeatherService(client)

    def geocode(args: GeocodeArgs) -> dict[str, Any]:
        places = geocoder.search(args.place, country_code=args.country_code, limit=3)
        if not places:
            raise ToolError(f"No place called '{args.place}' was found.")
        return {
            "matches": [
                {
                    "label": p.label,
                    "latitude": p.latitude,
                    "longitude": p.longitude,
                    "timezone": p.timezone,
                }
                for p in places
            ]
        }

    def get_weather(args: WeatherArgs) -> dict[str, Any]:
        report = weather.for_dates(args.latitude, args.longitude, args.start_date, args.end_date)
        return weather_payload(report)

    def get_distance(args: DistanceArgs) -> Any:
        return distance_report(
            args.from_latitude, args.from_longitude, args.to_latitude, args.to_longitude
        )

    today = weather.today()
    horizon = (today + timedelta(days=15)).isoformat()
    return ToolRegistry(
        [
            Tool(
                name="geocode",
                description=(
                    "Look up the coordinates of a place by name. Returns up to three matches; "
                    "pick the one in the right country."
                ),
                args_model=GeocodeArgs,
                handler=geocode,
            ),
            Tool(
                name="get_weather",
                description=(
                    "Daily weather for a location and date range. Dates up to "
                    f"{horizon} get a forecast; later dates get typical weather from past "
                    "years. Days marked plan_indoor_alternatives are likely to be wet."
                ),
                args_model=WeatherArgs,
                handler=get_weather,
            ),
            Tool(
                name="get_distance",
                description=(
                    "Straight-line distance between two coordinates with rough walking, "
                    "public transport and driving times. Use it to keep each day's stops "
                    "close together."
                ),
                args_model=DistanceArgs,
                handler=get_distance,
            ),
        ]
    )


def build_tools_client(data_dir: Path) -> OpenMeteoClient:
    """An Open-Meteo client caching under ``<data_dir>/http-cache/open-meteo``."""
    return OpenMeteoClient(cache=JsonCache(Path(data_dir) / "http-cache" / "open-meteo"))
