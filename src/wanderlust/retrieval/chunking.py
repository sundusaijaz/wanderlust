"""Split parsed articles into small, self-describing chunks for retrieval."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum

from pydantic import BaseModel

from wanderlust.wikivoyage.models import Article, Attribution, Listing, ListingType, Section

OVERVIEW = "Overview"
DEFAULT_MAX_CHARS = 800

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkKind(StrEnum):
    TEXT = "text"
    LISTING = "listing"


class Chunk(BaseModel):
    """A passage that can be searched and cited on its own."""

    id: str
    article: str
    section: str
    parent_section: str | None = None
    kind: ChunkKind
    text: str
    attribution: Attribution
    listing_name: str | None = None
    listing_type: ListingType | None = None
    lat: float | None = None
    lon: float | None = None

    @property
    def breadcrumb(self) -> str:
        """Where the chunk lives, e.g. 'Sintra > See > Other sites'."""
        parts = [self.article, self.parent_section, self.section]
        return " > ".join(part for part in parts if part)

    @property
    def search_text(self) -> str:
        """Text used for indexing: the breadcrumb gives each chunk its context."""
        return f"{self.breadcrumb}\n{self.text}"


def chunk_article(article: Article, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Turn an article into text chunks (per section) and one chunk per listing."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")

    chunks: list[Chunk] = []
    if article.summary:
        overview = Attribution(title=article.title, url=article.url)
        for n, piece in enumerate(split_text(article.summary, max_chars)):
            chunks.append(_text_chunk(article, OVERVIEW, None, piece, overview, n))

    for section in article.sections:
        for n, piece in enumerate(split_text(section.text, max_chars)):
            chunks.append(
                _text_chunk(article, section.title, section.parent, piece, section.attribution, n)
            )
        for n, listing in enumerate(section.listings):
            chunks.append(_listing_chunk(article, section, listing, n))
    return chunks


def split_text(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most ``max_chars``.

    Long paragraphs are split at sentence boundaries, and sentences that are
    still too long are split at word boundaries.
    """
    units: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n")):
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            units.extend(_split_words(sentence, max_chars))

    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = unit
    if current:
        pieces.append(current)
    return pieces


def _split_words(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = word[:max_chars]
    if current:
        parts.append(current)
    return parts


def _listing_text(listing: Listing) -> str:
    name = listing.name + (f" ({listing.alt})" if listing.alt else "")
    lines = [f"{name}: {listing.description}" if listing.description else name]
    for label, value in (
        ("Price", listing.price),
        ("Hours", listing.hours),
        ("Address", listing.address),
        ("Directions", listing.directions),
    ):
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _chunk_id(article: Article, *parts: object) -> str:
    key = "|".join([article.language, article.title, *map(str, parts)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _text_chunk(
    article: Article,
    section: str,
    parent: str | None,
    text: str,
    attribution: Attribution,
    n: int,
) -> Chunk:
    return Chunk(
        id=_chunk_id(article, section, parent, ChunkKind.TEXT, n),
        article=article.title,
        section=section,
        parent_section=parent,
        kind=ChunkKind.TEXT,
        text=text,
        attribution=attribution,
    )


def _listing_chunk(article: Article, section: Section, listing: Listing, n: int) -> Chunk:
    return Chunk(
        id=_chunk_id(article, section.title, section.parent, ChunkKind.LISTING, n),
        article=article.title,
        section=section.title,
        parent_section=section.parent,
        kind=ChunkKind.LISTING,
        text=_listing_text(listing),
        attribution=section.attribution,
        listing_name=listing.name,
        listing_type=listing.type,
        lat=listing.lat,
        lon=listing.lon,
    )
