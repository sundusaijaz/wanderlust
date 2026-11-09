"""Data models for Wikivoyage articles."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import quote

from pydantic import BaseModel, Field

WIKIVOYAGE_LICENSE = "CC BY-SA 4.0"
WIKIVOYAGE_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def article_url(language: str, title: str, anchor: str | None = None) -> str:
    """Build the public URL of a Wikivoyage article, optionally pointing at a section."""
    url = f"https://{language}.wikivoyage.org/wiki/{quote(title.replace(' ', '_'))}"
    if anchor:
        url += "#" + quote(anchor.replace(" ", "_"))
    return url


class ListingType(StrEnum):
    """Kinds of listing used on Wikivoyage (see, do, buy, eat, drink, sleep, go)."""

    SEE = "see"
    DO = "do"
    BUY = "buy"
    EAT = "eat"
    DRINK = "drink"
    SLEEP = "sleep"
    GO = "go"
    MARKER = "marker"
    OTHER = "other"


class Attribution(BaseModel):
    """Where a piece of content came from and under which license."""

    title: str
    url: str
    license: str = WIKIVOYAGE_LICENSE
    license_url: str = WIKIVOYAGE_LICENSE_URL


class Listing(BaseModel):
    """A single place from a Wikivoyage listing template."""

    type: ListingType
    name: str
    alt: str | None = None
    description: str = ""
    address: str | None = None
    directions: str | None = None
    phone: str | None = None
    hours: str | None = None
    price: str | None = None
    url: str | None = None
    lat: float | None = None
    lon: float | None = None

    @property
    def has_coordinates(self) -> bool:
        return self.lat is not None and self.lon is not None


class Section(BaseModel):
    """One heading of an article with its plain text and listings."""

    title: str
    level: int = Field(ge=2, le=3)
    parent: str | None = Field(
        default=None, description="Title of the enclosing level-2 section, for subsections."
    )
    text: str = ""
    listings: list[Listing] = Field(default_factory=list)
    attribution: Attribution

    @property
    def is_empty(self) -> bool:
        return not self.text and not self.listings


class Article(BaseModel):
    """A parsed Wikivoyage destination article."""

    title: str
    language: str
    page_id: int
    revision_id: int
    summary: str = Field(default="", description="The lead text before the first heading.")
    sections: list[Section] = Field(default_factory=list)
    districts: list[str] = Field(
        default_factory=list,
        description="Titles of district sub-articles (e.g. 'Lisbon/Baixa') for big cities.",
    )
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    license: str = WIKIVOYAGE_LICENSE
    license_url: str = WIKIVOYAGE_LICENSE_URL

    @property
    def url(self) -> str:
        return article_url(self.language, self.title)

    @property
    def listings(self) -> list[Listing]:
        return [listing for section in self.sections for listing in section.listings]

    def section(self, title: str) -> Section | None:
        """Find a section by title, ignoring case and surrounding whitespace."""
        wanted = title.strip().casefold()
        return next((s for s in self.sections if s.title.casefold() == wanted), None)

    def subsections(self, parent: str) -> list[Section]:
        """Level-3 sections nested under the given level-2 section."""
        wanted = parent.strip().casefold()
        return [s for s in self.sections if s.parent and s.parent.casefold() == wanted]
