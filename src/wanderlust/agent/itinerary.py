"""The structured itinerary the planning agent must produce.

The same models are the JSON Schema of the ``submit_itinerary`` tool and the
validator for what the model submits, so a plan is either well-formed or sent
back to the model with the exact problems listed.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from enum import StrEnum
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wanderlust.qa.citations import cited_numbers


class TimeOfDay(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Stop(_Strict):
    time_of_day: TimeOfDay
    name: str = Field(min_length=1, description="The place or activity, as named in the guide.")
    description: str = Field(
        min_length=1, description="One or two sentences: what to do there and why it fits."
    )
    sources: list[int] = Field(
        min_length=1,
        description="Numbers of the guide passages that mention this stop, e.g. [3, 7].",
    )
    indoor: bool = Field(default=False, description="True if it works on a rainy day.")


class DayPlan(_Strict):
    day: int = Field(ge=1, description="1 for the first day, 2 for the second, ...")
    date: dt.date | None = Field(
        default=None, description="The calendar date (YYYY-MM-DD), if the trip has dates."
    )
    base: str = Field(min_length=1, description="Where the traveller sleeps that night.")
    title: str = Field(min_length=1, description="A short theme, e.g. 'Old town and the castle'.")
    weather: str | None = Field(
        default=None, description="Expected weather from get_weather, in a few words."
    )
    stops: list[Stop] = Field(min_length=1)
    getting_around: str | None = Field(
        default=None, description="How to move between the stops, or to the next base."
    )

    @model_validator(mode="after")
    def _stops_in_order(self) -> DayPlan:
        order = list(TimeOfDay)
        positions = [order.index(stop.time_of_day) for stop in self.stops]
        if positions != sorted(positions):
            raise ValueError("stops must be ordered morning, afternoon, evening")
        return self


class Itinerary(_Strict):
    title: str = Field(min_length=1, description="e.g. '4 days of food and history in Lisbon'.")
    summary: str = Field(min_length=1, description="Two or three sentences about the trip.")
    days: list[DayPlan] = Field(min_length=1, max_length=21)
    tips: list[str] = Field(
        default_factory=list, description="Practical tips from the guides, each with citations."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Anything you had to assume, e.g. 'Dates taken as 10-13 March 2027'.",
    )

    @model_validator(mode="after")
    def _days_are_consecutive(self) -> Itinerary:
        numbers = [day.day for day in self.days]
        if numbers != list(range(1, len(self.days) + 1)):
            raise ValueError("days must be numbered 1, 2, 3, ... in order")
        dates = [day.date for day in self.days]
        if any(dates) and not all(dates):
            raise ValueError("give a date for every day or for none")
        if all(dates):
            for earlier, later in pairwise(dates):
                assert earlier is not None and later is not None
                if later - earlier != dt.timedelta(days=1):
                    raise ValueError("dates must be consecutive, one per day")
        return self

    @property
    def start_date(self) -> dt.date | None:
        return self.days[0].date

    def stops(self) -> Iterator[tuple[DayPlan, Stop]]:
        for day in self.days:
            for stop in day.stops:
                yield day, stop

    def cited_numbers(self) -> list[int]:
        """Every passage number the itinerary cites, in order of first use, without repeats.

        Tips may cite passages inline as [n]; those count too.
        """
        numbers: dict[int, None] = {}
        for _, stop in self.stops():
            for number in stop.sources:
                numbers.setdefault(number, None)
        for tip in self.tips:
            for number in cited_numbers(tip):
                numbers.setdefault(number, None)
        return list(numbers)
