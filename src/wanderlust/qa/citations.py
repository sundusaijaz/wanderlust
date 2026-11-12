"""Find and check the [n] citations in an answer."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from wanderlust.qa.sources import Source

# Matches [3], [1, 2] and [1,2,5]; adjacent groups like [1][3] match separately.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class CitationReport:
    cited: list[Source]
    """Sources the answer cites, in order of first citation."""
    unknown: list[int]
    """Cited numbers that match no source (the model made them up)."""
    uncited: list[Source]
    """Sources that were provided but not cited."""


def cited_numbers(text: str) -> list[int]:
    """All citation numbers in order of first appearance, without repeats."""
    numbers: dict[int, None] = {}
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            numbers.setdefault(int(part), None)
    return list(numbers)


def check_citations(text: str, sources: Sequence[Source]) -> CitationReport:
    by_number = {source.number: source for source in sources}
    numbers = cited_numbers(text)
    cited = [by_number[n] for n in numbers if n in by_number]
    unknown = [n for n in numbers if n not in by_number]
    cited_ids = {source.number for source in cited}
    uncited = [source for source in sources if source.number not in cited_ids]
    return CitationReport(cited=cited, unknown=unknown, uncited=uncited)
